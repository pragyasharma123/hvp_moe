# Author: Piyush Vyas
# modified by Chuks Ogbogu
# modified by Gaurav Narang for MOO 
import time
import torch
import argparse
import numpy as np
import utils as utils

import torchvision.datasets as datasets
import torchvision.transforms as transforms

import tf_utils
import sklearn.metrics
from models import GCN
from torch.utils.tensorboard import SummaryWriter
import os
from matplotlib import pyplot as plt
#from hessian_eigenthings import compute_hessian_eigenthings
from pytorch_hessian_eigenthings.hessian_eigenthings import compute_hessian_eigenthings
from noise_utils import *
from OU_moo_codebase.sim_ou_1 import * #offline_moo_optimization
#from noise_utils import add_gaussian_noise_adj
from online_ou_prediction.code.online_OU_GNN import *

import scipy.sparse as sp
import json

from args import parse_args
args = parse_args()

# Reproducibility
np.random.seed(0)
torch.manual_seed(0)

# Tensorboard Writer
writer = SummaryWriter('runs/' + args.dataset + '/' + str(args.exp_num))

# Settings based on ClusterGCN's Table 4. of section Experiments for different datasets
if args.dataset == 'amazon2M' and args.layers == 4: 
    _in = [100, args.hidden, args.hidden, args.hidden]
    _out = [args.hidden, args.hidden, args.hidden, 47]
if args.dataset == 'amazon2M' and args.layers == 9: 
    _in = [100, args.hidden, args.hidden, args.hidden, args.hidden, args.hidden, args.hidden, args.hidden, args.hidden]
    _out = [args.hidden, args.hidden, args.hidden, args.hidden, args.hidden, args.hidden, args.hidden, args.hidden, 47]
elif args.dataset == 'reddit' and args.layers == 4: 
        _in = [602, args.hidden, args.hidden, args.hidden]
        _out = [args.hidden, args.hidden, args.hidden, 41]
elif args.dataset == 'reddit' and args.layers == 9: 
    _in = [602, args.hidden, args.hidden, args.hidden, args.hidden, args.hidden, args.hidden, args.hidden, args.hidden]
    _out = [args.hidden, args.hidden, args.hidden, args.hidden, args.hidden, args.hidden, args.hidden, args.hidden, 41]
elif args.dataset == 'ppi' and args.layers == 4: 
    _in = [50, args.hidden,  args.hidden, args.hidden]
    _out = [args.hidden,  args.hidden, args.hidden, 121]
elif args.dataset == 'ppi' and args.layers == 9: 
    _in = [50, args.hidden,  args.hidden, args.hidden, args.hidden, args.hidden, args.hidden, args.hidden, args.hidden]
    _out = [args.hidden,  args.hidden, args.hidden, args.hidden, args.hidden, args.hidden, args.hidden, args.hidden, 121]

elif args.dataset == 'flickr' and args.layers == 5: 
    _in = [500, args.hidden, args.hidden, args.hidden, args.hidden]
    _out = [args.hidden, args.hidden, args.hidden, args.hidden, 7]

elif args.dataset == 'yelp' and args.layers == 4: 
    _in = [300, args.hidden, args.hidden, args.hidden]
    _out = [args.hidden, args.hidden, args.hidden, 100]
elif args.dataset == 'yelp' and args.layers == 9: 
    _in = [300, args.hidden,  args.hidden, args.hidden, args.hidden, args.hidden, args.hidden, args.hidden, args.hidden]
    _out = [args.hidden,  args.hidden, args.hidden, args.hidden, args.hidden, args.hidden, args.hidden, args.hidden, 100]
    

def save_checkpoint(state, filename):
    torch.save(state, filename)
    #shutil.copyfile(filename, 'model_best.pth.tar')

def load_data(data_prefix, dataset_str, precalc):
    """Return the required data formats for GCN models."""
    (num_data, train_adj, full_adj, feats, train_feats, test_feats, labels,
    train_data, val_data,
    test_data) = tf_utils.load_graphsage_data(data_prefix, dataset_str)
    visible_data = train_data
 
    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_data, :] = labels[train_data, :]
    y_val[val_data, :] = labels[val_data, :]
    y_test[test_data, :] = labels[test_data, :]
 
    train_mask = tf_utils.sample_mask(train_data, labels.shape[0])
    val_mask = tf_utils.sample_mask(val_data, labels.shape[0])
    test_mask = tf_utils.sample_mask(test_data, labels.shape[0])
 
    if precalc:
        train_feats = train_adj.dot(feats)
        #print("train_feats", np.shape(train_feats))
        #print("feats", np.shape(feats))
        #train_feats = np.hstack((train_feats, feats))
        #print("train_feats after hstack", np.shape(train_feats))
        test_feats = full_adj.dot(feats)
        test_feats = np.hstack((test_feats, feats))
    #print("train_adj ", np.shape(train_adj))
    #print("feats ", np.shape(feats))
    #print("full_adj", np.shape(full_adj))
    return (num_data, full_adj, feats, labels, train_adj, train_feats, train_data,
            val_data, test_data, y_train, y_val, y_test, train_mask, val_mask, test_mask)



def load_yelp_data(data_path):
    """
    Loads Yelp-style GraphSAGE-format dataset.
    
    Returns:
        num_data: int, number of nodes
        train_adj: scipy sparse matrix of training subgraph
        full_adj: scipy sparse matrix of full graph
        feats: np.ndarray [num_nodes, num_features]
        train_feats: np.ndarray, features for training nodes
        test_feats: np.ndarray, features for test nodes
        labels: np.ndarray [num_nodes, num_classes or num_labels]
        train_data: list of training node indices
        val_data: list of validation node indices
        test_data: list of test node indices
    """
    
    # Load adjacency matrices
    full_adj = sp.load_npz(f"{data_path}/adj_full.npz")
    train_adj = sp.load_npz(f"{data_path}/adj_train.npz")
    
    # Load features
    feats = np.load(f"{data_path}/feats.npy").astype(np.float32) # float not double
    
    # Load class map
    with open(f"{data_path}/class_map.json") as f:
        class_map = json.load(f)
    
    # Load train/val/test splits
    with open(f"{data_path}/role.json") as f:
        role = json.load(f)
    
    # Sort class map keys and convert to labels array
    num_data = feats.shape[0]
    if isinstance(class_map[list(class_map.keys())[0]], list):  # multi-label
        num_classes = len(class_map[list(class_map.keys())[0]])
        labels = np.zeros((num_data, num_classes), dtype=np.float32)
    else:  # single-label
        labels = np.zeros((num_data,), dtype=np.int64)

    for key, val in class_map.items():
        node_id = int(key)
        labels[node_id] = np.array(val)

    # Train/val/test splits
    train_data = role['tr']
    val_data = role['va']
    test_data = role['te']
    
    # Extract features for train/test nodes
    train_feats = feats[train_data]
    test_feats = feats[test_data]

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_data, :] = labels[train_data, :]
    y_val[val_data, :] = labels[val_data, :]
    y_test[test_data, :] = labels[test_data, :]

    train_mask = tf_utils.sample_mask(train_data, labels.shape[0])
    val_mask = tf_utils.sample_mask(val_data, labels.shape[0])
    test_mask = tf_utils.sample_mask(test_data, labels.shape[0])

    return (num_data, full_adj, feats, labels, train_adj, train_feats, train_data, 
              val_data, test_data, y_train, y_val, y_test, train_mask, val_mask, test_mask)


def train(model, criterion, optimizer, features, adj, labels, dataset):
        optimizer.zero_grad()
        
        #features = torch.from_numpy(features).cuda()
        #features.requires_grad_(True)  # Node features need to track gradients if involved
        # print size features
        #print('input size', features.size())
        #labels = torch.LongTensor(labels).cuda()
        # print size labels
        #print('target size', labels.size())
        features = torch.FloatTensor(np.array(features)).cuda() # conversion of np.array to tensor is faster
        labels = torch.LongTensor(np.array(labels)).cuda()
        
        # Adj -> Torch Sparse Tensor
        i = torch.LongTensor(adj[0]) # indices
        v = torch.FloatTensor(adj[1]) # values
        #adj = torch.sparse.FloatTensor(i.t(), v, adj[2]).cuda()
        # use this : torch.sparse_coo_tensor(indices, values, shape, dtype=, device=)
        adj = torch.sparse_coo_tensor(i.t(), v, adj[2]).cuda()

        #print('adj size sparse', adj.size())
        #exit()
        #Apply noise to adj here. # only if you want noise in training
        #convert to dense, apply noise, and convert back to sparse as above (torch.sparse))

        output = model(adj, features)
        loss = criterion(output, torch.max(labels, 1)[1])
        #print('loss', loss)

        #Verify which parameters are involved in the computation graph
        #for param in model.parameters():
        #    print(param.requires_grad, param.grad_fn) # prints True, None : it should not be none for all
        
        '''
        print('grad func ?',loss.grad_fn)  # Should return a grad function if connected properly # it does

        for name, p in model.named_parameters():
            if 'weight' in name and 'layer_norm' not in name:
                weights = p
                grad = torch.autograd.grad(loss, weights, create_graph=True)
                print(grad)
                print(grad[0])
                exit()
        '''

        #grad_dict = torch.autograd.grad(
        #        loss, model.parameters(), create_graph=True#, allow_unused=True
        #    )
        

        loss.backward()

        #for name, param in model.named_parameters():
        #    print(f"{name}: {param.grad}")  # Check gradients after backward # looks fine

        optimizer.step()
        pred = output.argmax(dim=1, keepdim=True)
        labels = labels.argmax(dim=1, keepdim=True)
        
        accuracy = (pred == labels).float().sum()
        total = len(labels)
        #exit()
    
        return loss, accuracy, total


@torch.no_grad()
def test(row_ou, col_ou, drift_time, temperature, features, adj, labels, mask):
        # load pre-trained model
        modelfile = 'saves/'+args.dataset+'/'+args.pre_trained_model
        model = torch.load(f"{modelfile}")
        model.eval()

        for name, p in model.named_parameters():
            if 'weight' in name and 'layer_norm' not in name:
                weights = p
                print('pre-noise weights',weights)

        # splitting temperature array into adj and weights
        temp_adj = temperature[0]
        temperature_weights=np.delete(temperature, 0)[:args.layers] # remove 0th index and keep first n elements
        row_ou_adj = row_ou[0]
        col_ou_adj = col_ou[0]
        row_ou_weights = np.delete(row_ou, 0)[:args.layers]  # remove 0th index and keep first n elements
        col_ou_weights = np.delete(col_ou, 0)[:args.layers]  # remove 0th index and keep first n elements
        ###

        #print('temp_adj',temp_adj)
        #print('temperature_weights',temperature_weights)
        #print('row_ou_adj', row_ou_adj)
        #print('col_ou_adj',col_ou_adj)
        #print('row_ou_weights ',row_ou_weights )
        #print('col_ou_weights ',col_ou_weights )

        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        print('=> device', device)
       
        #############################
        features = torch.FloatTensor(np.array(features)).to(device) # conversion of np.array to tensor is faster
        labels = torch.LongTensor(np.array(labels)).to(device)
        #exit()
        
        #if device == 'cpu':
        adj = adj[0]
        features = features[0]
        labels = labels[0]
        mask = mask[0]
            
        # Adj -> Torch Sparse Tensor
        #print(adj[0])
        i = torch.LongTensor(adj[0]) # indices # LongTensor gave depracated warning
        
        v = torch.FloatTensor(adj[1]) # values
        #adj = torch.sparse.FloatTensor(i.t(), v, adj[2]).to(device)
        adj = torch.sparse_coo_tensor(i.t(), v, adj[2]).to(device)
        #print('sparse format adj', adj)
                
        # apply noise during inference here.
        if args.enable_noise_adj:
            # add noise to adj
            print("=> add noise to adj matrix")
            #adj=add_gaussian_noise_adj(adj, device)  
            #adj=add_noise_adj(adj, drift_time, temp_adj, device)
            adj=add_noise_adj(adj, drift_time, temp_adj, device, row_ou_adj, col_ou_adj)
            adj=adj.to_sparse()
        if args.enable_noise_weights:      
            print('=> add layer-wise noise')            
            model, delta_w = weights_to_noisy_weights(model, temperature_weights, drift_time, row_ou_weights, col_ou_weights)

            print('delta_w', delta_w)
        # already passing in noisy model, so commented here
        #if args.enable_noise_weights:      
        #    print('=> add layer-wise noise')
        #    model=add_gaussian_noise_layerwise(model, noise_config, device)
        ### end of noise handling
        
        print('=> compute output')
        #print(adj)
        #print(features)
        model.cuda()

        for name, p in model.named_parameters():
            if 'weight' in name and 'layer_norm' not in name:
                weights = p
                print('after-noise weights',weights)

        output = model(adj, features)
        
        pred = output[mask].argmax(dim=1, keepdim=True)
        labels = torch.max(labels[mask], 1)[1]
        return sklearn.metrics.f1_score(labels.cpu().numpy(), pred.cpu().numpy(), average='micro')



def graph_data_processing(start):
    # use this for other datasets
    if args.dataset == "yelp":
        # use this for yelp
        (N, _adj, _feats, _labels, train_adj, train_feats, train_nodes, val_nodes, test_nodes, y_train, y_val, y_test, train_mask, val_mask, test_mask) = load_yelp_data(os.getcwd()+'/'+'../datasets/yelp')
    else:
        (N, _adj, _feats, _labels, train_adj, train_feats, train_nodes, val_nodes, test_nodes, y_train, y_val, y_test, train_mask, val_mask, test_mask) = load_data(os.getcwd()+'/'+'../datasets', args.dataset, True)
    
    
    print('Loaded data in {:.2f} seconds!'.format(time.time() - start))
    
    # Prepare Train Data
    start = time.time()
    _, parts = utils.partition_graph(train_adj, train_nodes, args.num_clusters_train)
    parts = [np.array(pt) for pt in parts]
    #train_features, train_support, y_train = utils.preprocess_multicluster(train_adj, parts, train_feats, y_train, args.num_clusters_train, args.batch_size)    

    if args.dataset == "yelp":
        train_features, train_support, y_train, train_mask = tf_utils.preprocess_multicluster(train_adj, parts, _feats, y_train, train_mask, args.num_clusters_train, args.batch_size)
    else:
        train_features, train_support, y_train, train_mask = tf_utils.preprocess_multicluster(train_adj, parts, train_feats, y_train, train_mask, args.num_clusters_train, args.batch_size) # requires _feats to match global indexing
     
    print('Train Data pre-processed in {:.2f} seconds!'.format(time.time() - start))
    
    # Prepare Test Data
    if args.test == 1:    
        y_test, test_mask = y_val, val_mask
        start = time.time()
        _, test_features, test_support, y_test, test_mask = utils.preprocess(_adj, _feats, y_test, np.arange(N), args.num_clusters_test, test_mask) 
        print('Test Data pre-processed in {:.2f} seconds!'.format(time.time() - start))

    if args.compute_hessian: # took this out of test loop to use for training as well
            
        samples = [(x_i, y_i) for x_i, y_i in zip(train_features[args.batch], y_train[args.batch])] # passing only [0] index
        # full dataset
        dataloader = torch.utils.data.DataLoader(samples, batch_size=len(y_train[args.batch]), shuffle=False)
    
    return train_support, train_features, dataloader

def image_data_processing():
    #CNN Data Loader
    transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),

                ])
    transform_test = transforms.Compose([
    transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    # ADD here for CNN and ViT  
    # output should be subgraph_batch=0, hess_dataloader, criterion
    if args.dataset=="cifar10":
        from archs.cifar10 import vgg,resnet,AlexNet,googlenet,mobilenet, dualpath, densenet#, vit

        traindataset = datasets.CIFAR10(f"{os.getcwd()}/saves/{args.arch}/{args.dataset}", train=True, download=True,transform=transform_train)
        testdataset = datasets.CIFAR10(f"{os.getcwd()}/saves/{args.arch}/{args.dataset}", train=False, transform=transform_test)
        train_loader=torch.utils.data.DataLoader(traindataset, batch_size=args.batch_size, shuffle=False, pin_memory=True,num_workers=2)
        test_loader=torch.utils.data.DataLoader(testdataset, batch_size=100, shuffle=False, pin_memory=True,num_workers=2)
        

    elif args.dataset=="cifar100":
        from archs.cifar100 import vgg,resnet,AlexNet,googlenet,mobilenet, dualpath, densenet

        traindataset = datasets.CIFAR100(f"{os.getcwd()}/saves/{args.arch}/{args.dataset}", train=True, download=True,transform=transform_train)
        testdataset = datasets.CIFAR100(f"{os.getcwd()}/saves/{args.arch}/{args.dataset}", train=False, transform=transform_test)
        train_loader=torch.utils.data.DataLoader(traindataset, batch_size=args.batch_size, shuffle=False, pin_memory=True,num_workers=2)
        test_loader=torch.utils.data.DataLoader(testdataset, batch_size=100, shuffle=False, pin_memory=True,num_workers=2)

    elif args.dataset=="TinyImageNet":
        from archs.TinyImageNet import vgg, resnet, vit

        data_dir = f'{os.getcwd()}/datasets/tiny-imagenet-200/'
        num_workers = {"train": 2, "val": 0, "test": 0}
        data_transforms = {
            "train": transforms.Compose([transforms.Resize((224, 224)),transforms.ToTensor(),]), #transform_train,
            "val": transforms.Compose([transforms.Resize((224, 224)),transforms.ToTensor(),]), #transform_test,
        }
        image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ["train", "val"]}
        dataloaders = {
            x: torch.utils.data.DataLoader(image_datasets[x], batch_size=100, shuffle=True, num_workers=num_workers[x])
            for x in ["train", "val"]
        }
        train_loader=dataloaders["train"]
        test_loader=dataloaders["val"]

    else:
        print("Wrong dataset choice...fix this")
        exit()       

    if args.task_type=="image": # images
        list_x=[]
        list_y=[]
        
        #for data, target in test_loader:            
        #    list_x.append(data)
        #    list_y.append(target)

        for batch_idx, (imgs, targets) in enumerate(train_loader):
            list_x.append(imgs)
            list_y.append(targets)
            #break
        

        hess_samples = [(x,y) for x,y in zip(list_x[0],list_y[0])] # passing only [0] index
    
        hess_dataloader = DataLoader(hess_samples, batch_size=len(list_y[0]), shuffle=False) #batch_size=len(data.y), #len(list_x)
        
        subgraph_batch=0
        #hess_dataloader= train_loader
        criterion = torch.nn.CrossEntropyLoss()
        
        modelfile = 'saves/'+args.arch+'/'+args.dataset+'/'+args.pre_trained_model
        model = torch.load(f"{modelfile}")

        print("Computing hessian score")
        eigenvals, eigenvecs = compute_hessian_eigenthings(
                                    model, # without noise - pre-trained
                                    hess_dataloader, #
                                    0, # adj matrix
                                    criterion, #
                                    args.num_eigenthings,
                                    mode=args.mode,
                                    # power_iter_steps=args.num_steps,
                                    max_possible_gpu_samples=args.max_samples,
                                    # momentum=args.momentum,
                                    full_dataset=args.full_dataset,
                                    use_gpu=True#use_cuda,
                                    )
        print("Eigenvecs:")
        print(eigenvecs)
        #print('eigenvecs shape',eigenvecs.shape)
        print("Eigenvals:")
        print(eigenvals)

    return subgraph_batch, hess_dataloader, eigenvecs, eigenvals

def main():    

    if args.runtime_ou==1 and args.enable_noise_adj==1:
        print("runtime OU : disable enable_noise_adj for kernel/layer senstivity through hessian")
        exit()        

    if args.runtime_ou==1 and args.enable_noise_weights==1:
        print("runtime OU : disable enable_noise_weights for kernel/layer senstivity through hessian")
        exit()

    #if args.accuracy_eval and not args.enable_noise_weights:
    #    print("for non-ideal accuracy calc., enable enable_noise_adj and enable_noise_weights")
    #    exit()
    
    if args.task_type=="graph":
        modelfile = 'saves/'+args.arch+'/'+args.dataset+'/'+args.pre_trained_model

        if not os.path.isfile(modelfile):
            print('pre-trained model not found')
            exit()
        else:
            print('pre-trained model (GCN/GAT/SAGE) exists')
    
    # Load data
    start = time.time()
    
    if args.task_type=="graph":
        eigenvecs, eigenvals = None, None # for graphs they will not be evaluated here
        train_support, train_features, dataloader = graph_data_processing(start)

    elif args.task_type=="image":
        if args.compute_hessian==0 and args.validation_phase==1:
            train_support, dataloader, eigenvecs, eigenvals = None, None, None, None
        else:
            train_support, dataloader, eigenvecs, eigenvals = image_data_processing()

    else:
        print("invalid task... fix task_type arguement")
        exit()

    # Loss Function # common for all tasks
    criterion = torch.nn.CrossEntropyLoss()

    # model # model is called in runtime_ou_optim function later # so its outside any if condition for the moment
    if args.task_type=="graph":
        model = GCN(fan_in=_in, fan_out=_out, layers=args.layers,  dropout=args.dropout, normalize=True, bias=False).float() #num_nodes=train_adj.shape[0],
        print(model)
        model.cuda()


    if args.task_type=="graph":
        # Optimization Algorithm
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        
        # Learning Rate Schedule    
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=int(args.num_clusters_train/args.batch_size), epochs=args.epochs+1, anneal_strategy='linear')
        model.train()

        # Train only for graph
        # no training in moo code, so test=1

    

        # Shuffle Batches
        batch_idxs = list(range(len(train_features)))
        print('len(batch_idxs)',len(batch_idxs))

        if not args.test: # training
            print("training")
        
            for epoch in range(args.epochs + 1):
                #print('epoch', epoch)
                np.random.shuffle(batch_idxs)
                avg_loss = 0
                total_accuracy = 0
                total_total = 0
                start = time.time()
                for batch in batch_idxs:
                    loss, accuracy, total = train(model.cuda(), criterion, optimizer, train_features[batch], train_support[batch], y_train[batch], dataset=args.dataset)
                    if args.lr_scheduler == 1:
                        scheduler.step()
                    avg_loss += loss.item()
                    total_accuracy += accuracy
                    total_total += total
                acc =  float(total_accuracy)/float(total_total)
                print(acc)
        
                # Write Train stats to tensorboard
                writer.add_scalar('time/train', time.time() - start, epoch)
                writer.add_scalar('loss/train', avg_loss/len(train_features), epoch)

                # inside train loop - experiment used to find hessian importance in each epoch
                '''
                # save the model
                filename = 'saves/'+args.dataset+'/'+'1_noise_checkpoint.pth.tar'
                save_checkpoint(state={
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }, filename=filename)

                # compute hessian
                #train_model = torch.load(f"{filename}") # doesnt work for this dict structure
                print("=> loading checkpoint '{}'".format(filename))
                checkpoint = torch.load(filename)
                model.load_state_dict(checkpoint['state_dict'])
                #optimizer.load_state_dict(checkpoint['optimizer']) # not needed
                print("=> loaded checkpoint '{}'"
                      .format(filename))
                print(model)
                #exit()
                #train_adj = train_support[args.batch]
                adj_h=train_support[args.batch] # subgraph
                #print('adj shape', np.shape(adj)) # prints (3,)
                
                # Adj -> Torch Sparse Tensor
                i = torch.LongTensor(adj_h[0]) # indices
                v = torch.FloatTensor(adj_h[1]) # values
                #adj = torch.sparse.FloatTensor(i.t(), v, adj[2]).cuda()
                adj_h = torch.sparse_coo_tensor(i.t(), v, adj_h[2]).cuda()

                print("Computing hessian score")
                eigenvals, eigenvecs = compute_hessian_eigenthings(
                                    model, # without noise
                                    dataloader, #
                                    adj_h,#train_support[0], #
                                    criterion, #
                                    args.num_eigenthings,
                                    mode=args.mode,
                                    # power_iter_steps=args.num_steps,
                                    max_possible_gpu_samples=args.max_samples,
                                    # momentum=args.momentum,
                                    full_dataset=args.full_dataset,
                                    use_gpu=True#use_cuda,
                                    )
                print("Eigenvecs:")
                print(eigenvecs)
                #print('eigenvecs shape',eigenvecs.shape)
                print("Eigenvals:")
                print(eigenvals)

                delta_w=0            

                nonidealities=layer_score_func_training(epoch, model, delta_w, eigenvecs, eigenvals)
                '''

        
            # outside train loop
            filename = 'saves/'+args.dataset+'/'+'0_noise_checkpoint.pth.tar'
            save_checkpoint(state={
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, filename=filename)
        
    ########################################
    # run for all type of tasks
    if args.test:   # inference     # for graphs, images, etc.
        if args.runtime_ou==0:
            if not args.accuracy_eval:
                print('running MOO')
                offline_moo_optimization(10**0, train_support, dataloader, criterion, eigenvecs, eigenvals ) # updated arguements on july 14, 2025
            else:
                print('accuracy eval mode starting...')

                temperature=np.full((args.layers+1), 20, dtype=int)#[300,300,300,300,300]
                #temperature=[1,1,20,20,1]
                ou_vals=[4]#,8,16]#,32,128]
                drift_time=[0,2,4,8]
                for i in range(len(ou_vals)):
                    for j in range(len(drift_time)):
                        row_ou=np.full((args.layers+1), ou_vals[i], dtype=int) 
                        col_ou=np.full((args.layers+1), ou_vals[i], dtype=int)

                        print("CASE:", ou_vals[i], drift_time[j])
                        
                        nonideal_f1 = test(row_ou, col_ou, 10**drift_time[j], temperature, test_features, test_support, y_test, test_mask)
                        print('nonideal_f1 %: {:.4f}'.format(nonideal_f1*100))
                        #exit()

        else:            
            print('running online OU optim')
            runtime_ou_optim(model, train_support, dataloader, criterion) # may only work for graph (model is input, fix later)
            
                

        '''
        # check this here # load the model outside for loop and compute ideal test accuracy 
        print("=> loading checkpoint '{}'".format(modelfile))
        checkpoint = torch.load(modelfile)
        model.load_state_dict(checkpoint['state_dict'])
        #optimizer.load_state_dict(checkpoint['optimizer']) # not needed
        print("=> loaded checkpoint '{}'"
                  .format(modelfile))
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")

        drift_time=10**0 # 1 sec
        temp_adj=300 # ref temp
        ideal_f1 = test(drift_time, temp_adj, noise_config, model.to(device), test_features, test_support, y_test, test_mask, device=device)
        print('ideal_f1: {:.4f}'.format(ideal_f1)) 
        '''
     
        
if __name__ == '__main__':
    main()