"""
This module defines a linear operator to compute the hessian-vector product
for a given pytorch model using subsampled data.
"""

from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data


import pytorch_hessian_eigenthings.hessian_eigenthings.utils as utils

from pytorch_hessian_eigenthings.hessian_eigenthings.operator import Operator
#from noise_utils import add_gaussian_noise_layerwise
#from noise_utils import add_gaussian_noise_adj
from args import parse_args
args = parse_args()

class HVPOperator(Operator):
    """
    Use PyTorch autograd for Hessian Vec product calculation
    model:  PyTorch network to compute hessian for
    dataloader: pytorch dataloader that we get examples from to compute grads
    loss:   Loss function to descend (e.g. F.cross_entropy)
    use_gpu: use cuda or not
    max_possible_gpu_samples: max number of examples per batch using all GPUs.
    """

    def __init__(
        self,
        model: nn.Module,
        dataloader: data.DataLoader,
        train_support,#: train_support, # not sure to put it here or not
        criterion: Callable[[torch.Tensor], torch.Tensor],
        use_gpu: bool = True,
        fp16: bool = False,
        full_dataset: bool = True,
        max_possible_gpu_samples: int = 256,
    ):
        #size = int(sum(p.numel() for p in model.parameters()))
        if args.arch=="GCN":
            size = int(sum(p.numel() for name, p in model.named_parameters() if 'weight' in name and 'layer_norm' not in name))
        elif args.arch=="GAT":
            size = int(sum(p.numel() for name, p in model.named_parameters() if 'conv' in name and 'weight' in name and 'layer_norm' not in name))
        elif args.arch=="SAGE":
            size = int(sum(p.numel() for name, p in model.named_parameters() if 'conv' in name and 'weight' in name and 'lin_r' not in name))
        else: # vgg/resnet/vit etc.
            size = int(sum(p.numel() for name, p in model.named_parameters() if 'conv' in name and 'weight' in name and 'norm' not in name or 'linear' in name and 'bias' not in name))
        
        super(HVPOperator, self).__init__(size)
        self.grad_vec = torch.zeros(size)
        self.model = model
        if use_gpu:
            self.model = self.model.cuda()
        self.dataloader = dataloader
        # Make a copy since we will go over it a bunch
        self.dataloader_iter = iter(dataloader)
        self.train_support= train_support # ADD
        self.criterion = criterion
        self.use_gpu = use_gpu
        self.fp16 = fp16
        self.full_dataset = full_dataset
        self.max_possible_gpu_samples = max_possible_gpu_samples

        if not hasattr(self.dataloader, '__len__') and self.full_dataset:
            raise ValueError("For full-dataset averaging, dataloader must have '__len__'")

    def apply(self, vec: torch.Tensor):
        """
        Returns H*vec where H is the hessian of the loss w.r.t.
        the vectorized model parameters
        """
        if self.full_dataset:
            return self._apply_full(vec)
        else:
            return self._apply_batch(vec)

    def _apply_batch(self, vec: torch.Tensor) -> torch.Tensor:
        """
        Computes the Hessian-vector product for a mini-batch from the dataset.
        """
        # compute original gradient, tracking computation graph
        self._zero_grad()
        grad_vec = self._prepare_grad()
        self._zero_grad()
        # take the second gradient
        # this is the derivative of <grad_vec, v> where <,> is an inner product.

        #hessian_vec_prod_dict = torch.autograd.grad(
        #    grad_vec, self.model.parameters(), grad_outputs=vec, only_inputs=True
        #)
        hessian_vec_prod_dict=[]
        '''
        for name, p in self.model.named_parameters():
            if 'weight' in name and 'layer_norm' not in name:
                weights = p
                hessian_vec_prod_dict.append(torch.autograd.grad(grad_vec, weights, grad_outputs=vec, only_inputs=True, retain_graph=True)[0])
        '''
        for name, p in self.model.named_parameters():
            if args.arch=="GCN":
                if 'weight' in name and 'layer_norm' not in name:
                    weights = p
                    hessian_vec_prod_dict.append(torch.autograd.grad(grad_vec, weights, grad_outputs=vec, only_inputs=True, retain_graph=True)[0])
            elif args.arch=="GAT":
                if 'conv' in name and 'weight' in name and 'layer_norm' not in name:
                    weights = p
                    hessian_vec_prod_dict.append(torch.autograd.grad(grad_vec, weights, grad_outputs=vec, only_inputs=True, retain_graph=True)[0])
            elif args.arch=="SAGE":
                if 'conv' in name and 'weight' in name and 'lin_r' not in name:
                    weights = p
                    hessian_vec_prod_dict.append(torch.autograd.grad(grad_vec, weights, grad_outputs=vec, only_inputs=True, retain_graph=True)[0])
            elif args.task_type=="image": #CNN
                if 'conv' in name and 'weight' in name and 'norm' not in name or 'linear' in name and 'bias' not in name:
                    weights = p
                    hessian_vec_prod_dict.append(torch.autograd.grad(grad_vec, weights, grad_outputs=vec, only_inputs=True, retain_graph=True)[0])


        # concatenate the results over the different components of the network
        hessian_vec_prod = torch.cat([g.contiguous().view(-1) for g in hessian_vec_prod_dict])
        hessian_vec_prod = utils.maybe_fp16(hessian_vec_prod, self.fp16)
        return hessian_vec_prod

    def _apply_full(self, vec: torch.Tensor) -> torch.Tensor:
        """
        Computes the Hessian-vector product averaged over all batches in the dataset.

        """
        n = len(self.dataloader)
        hessian_vec_prod = None
        for _ in range(n):
            if hessian_vec_prod is not None:
                hessian_vec_prod += self._apply_batch(vec)
            else:
                hessian_vec_prod = self._apply_batch(vec)
        hessian_vec_prod = hessian_vec_prod / n
        return hessian_vec_prod

    def _zero_grad(self):
        """
        Zeros out the gradient info for each parameter in the model
        """
        for p in self.model.parameters():
            if p.grad is not None:
                p.grad.data.zero_()

    def _prepare_grad(self) -> torch.Tensor:
        """
        Compute gradient w.r.t loss over all parameters and vectorize
        """
        try:
            all_inputs, all_targets = next(self.dataloader_iter)
        except StopIteration:
            self.dataloader_iter = iter(self.dataloader)
            all_inputs, all_targets = next(self.dataloader_iter)

        num_chunks = max(1, len(all_inputs) // self.max_possible_gpu_samples)

        grad_vec = None

        # This will do the "gradient chunking trick" to create micro-batches
        # when the batch size is larger than what will fit in memory.
        # WARNING: this may interact poorly with batch normalization.

        input_microbatches = all_inputs.chunk(num_chunks)
        #print(input_microbatches) # tensor ([ [ [ ... ... ...] ] ])
        #print('input_microbatches',len(input_microbatches)) # gives 1
        target_microbatches = all_targets.chunk(num_chunks)
        #print('target_microbatches',len(target_microbatches))
        #exit()
        for input, target in zip(input_microbatches, target_microbatches):
            if self.use_gpu:
                #input = input.cuda() # original
                #target = target.cuda() # original
                #print('=> use_gpu==1')
                #input = torch.FloatTensor(np.array(input)).cuda() # used earlier
                #print('input size', input.size())
                #print('input', input)
                #target = torch.LongTensor(np.array(target)).cuda() # used earlier
                #print('target size', target.size())

                if args.task_type=="image":
                    input=input.cuda()
                    target=target.cuda()
                elif args.task_type=="graph":
                    input = torch.FloatTensor(np.array(input)).cuda()
                    target = torch.LongTensor(np.array(target)).cuda()
                    adj=self.train_support.cuda()#[0] # giving 0th from func call in main
                
                #adj=self.train_support.cuda()#[0] # giving 0th from func call in main
                #print('adj shape', np.shape(adj)) # prints (3,)
                '''
                # Adj -> Torch Sparse Tensor
                i = torch.LongTensor(adj[0]) # indices
                v = torch.FloatTensor(adj[1]) # values
                #adj = torch.sparse.FloatTensor(i.t(), v, adj[2]).cuda()
                adj = torch.sparse_coo_tensor(i.t(), v, adj[2]).cuda()
                '''

                '''
                # code for padding Adj
                adj_dense=adj.to_dense().cuda()
                pad_size=int((1024-adj_dense.size()[0])/2)
                #adj=np.pad(adj, [(0, 1024-adj.size()[0]),(0,1024-adj.size()[0])], mode='constant', constant_values=0).cpu()
                zp=nn.ZeroPad2d((pad_size, pad_size+1, pad_size, pad_size+1)) # fix me later # works for odd shape size # put an odd-even condition
                adj_padded=zp(adj_dense)
                adj_sparse=adj_padded.to_sparse().cuda()
                print('adj_sparse', adj_sparse)
                print('adj size before padding', adj_dense.size())
                print('adj size sparse', adj_sparse.size())
                '''


            #output = self.model(input) # cnn
            #output = self.model(adj_sparse, input) # gnn (sparse)
            #print("adj_shape", adj.shape)
            #print("feats shape:", input.shape)
            #exit()
            if args.task_type=="graph":
                output = self.model(adj, input) # gnn
                #print('output', output)
                #loss = self.criterion(output, target) # cnn
                loss = self.criterion(output, torch.max(target, 1)[1]) # gnn
                #print('loss', loss)
            elif args.task_type=="image":
                output = self.model(input) # cnn
                loss = self.criterion(output, target) # cnn
            else:
                print("wrong task")
                exit()
            
            grad_dict=[]
            for name, p in self.model.named_parameters():
                '''
                if 'weight' in name and 'layer_norm' not in name:
                    weights = p
                    grad_dict.append(torch.autograd.grad(loss, weights, create_graph=True)[0])
                '''
                if args.arch=="GCN":
                   if 'weight' in name and 'layer_norm' not in name:
                        weights = p
                        grad_dict.append(torch.autograd.grad(loss, weights, create_graph=True)[0])                 
                elif args.arch=="GAT":
                    if 'conv' in name and 'weight' in name and 'layer_norm' not in name:
                        weights = p
                        grad_dict.append(torch.autograd.grad(loss, weights, create_graph=True)[0])
                elif args.arch=="SAGE":
                    if 'conv' in name and 'weight' in name and 'lin_r' not in name:
                        weights = p
                        grad_dict.append(torch.autograd.grad(loss, weights, create_graph=True)[0])
                else: # CNN
                    if 'conv' in name and 'weight' in name and 'norm' not in name or 'linear' in name and 'bias' not in name:
                        #print('Y')
                        #exit()
                        weights = p
                        grad_dict.append(torch.autograd.grad(loss, weights, create_graph=True)[0])


            
            #grad_dict = torch.autograd.grad(
            #    loss, self.model.parameters(), create_graph=True#, allow_unused=True
            #)
            if grad_vec is not None:
                grad_vec += torch.cat([g.contiguous().view(-1) for g in grad_dict])
            else:
                grad_vec = torch.cat([g.contiguous().view(-1) for g in grad_dict])
            grad_vec = utils.maybe_fp16(grad_vec, self.fp16)
        grad_vec /= num_chunks
        self.grad_vec = grad_vec
        return self.grad_vec
