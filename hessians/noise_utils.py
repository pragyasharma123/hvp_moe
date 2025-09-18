import time
import torch
import argparse
import numpy as np
import utils as utils
import sklearn.metrics
#from models import GCN
#from torch.utils.tensorboard import SummaryWriter
import os
from matplotlib import pyplot as plt
import math

from pytorch_hessian_eigenthings.hessian_eigenthings import compute_hessian_eigenthings

#from torch_geometric.utils import *#to_dense_adj

from args import parse_args

args = parse_args()

def compute_nonidealities(drift_time, temperature, row_ou, col_ou, train_support, dataloader, criterion, eigenvecs, eigenvals ):

    '''
    temperature= torch.randint(low=330, high=500, size=(args.layers,)) # will be an input
    print('temp Q', temperature)
    temp_adj=torch.randint(low=330, high=500, size=(1,))
    print('temp of adj matrix', temp_adj)
    drift_time=10**2 # will be an input
    '''

    # load pre-trained model
    if args.dataset=="cifar10" or args.dataset=="cifar100" or args.dataset=="TinyImageNet": # CNNs and Transformers
        modelfile = 'saves/'+args.arch+'/'+args.dataset+'/'+args.pre_trained_model
    else: # GNNs
        modelfile = 'saves/'+args.arch+'/'+args.dataset+'/'+args.pre_trained_model

    if os.path.isfile(modelfile):
    
        model = torch.load(f"{modelfile}")
        #print(model)
        #exit()
        if args.task_type=="graph":
            # splitting temperature array into adj and weights
            temp_adj = temperature[0]
            temperature_weights=np.delete(temperature, 0)[:args.layers] # remove 0th index and keep first n elements
            row_ou_adj = row_ou[0]
            col_ou_adj = col_ou[0]
            row_ou_weights = np.delete(row_ou, 0)[:args.layers]  # remove 0th index and keep first n elements
            col_ou_weights = np.delete(col_ou, 0)[:args.layers]  # remove 0th index and keep first n elements

            print('temp_adj',temp_adj)
            print('row_ou_adj', row_ou_adj)
            print('col_ou_adj',col_ou_adj)
        elif args.task_type=="image":
            #convert to numpy to address error in g_min_ni equation later: TypeError: unsupported operand type(s) for -: 'list' and 'int'
            temperature_weights=np.array(temperature[:args.layers])
            row_ou_weights = np.array(row_ou[:args.layers])
            col_ou_weights = np.array(col_ou[:args.layers])

        
        print('temperature_weights',temperature_weights)    
        print('row_ou_weights ',row_ou_weights )
        print('col_ou_weights ',col_ou_weights )

        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")

        if args.dataset=="ppi" or args.dataset=="reddit" or args.dataset=="ogbl_citation2":              #args.arch=="GAT":
            adj_h=train_support[args.batch] # subgraph # clustergcn
        elif args.dataset=="Reddit" or args.dataset=="ogbn_proteins":         #args.arch=="SAGE":
            adj_h=train_support # pytorch geometric
        elif args.dataset=="cifar10" or args.dataset=="cifar100" or args.dataset=="TinyImageNet":
            adj_h=0
        else:
            print("Error in noise_utils.py : wrong dataset choice..fix")
            exit()
        #print('adj shape', np.shape(adj)) # prints (3,)
                

        if args.task_type=="graph":
            if args.arch=="GCN":
                # Adj -> Torch Sparse Tensor
                i = torch.LongTensor(adj_h[0]) # indices
                v = torch.FloatTensor(adj_h[1]) # values
                adj_h = torch.sparse_coo_tensor(i.t(), v, adj_h[2]).cuda()

            if args.enable_noise_adj:
                # add noise to adj
                print("=> add noise to adj matrix")
                #adj_h=add_gaussian_noise_adj(adj_h, device)
                adj_h=add_noise_adj(adj_h, drift_time, temp_adj, device, row_ou_adj, col_ou_adj)
                if args.arch=="GCN":
                    adj_h=adj_h.to_sparse()  # clusterGCN
            
        if args.enable_noise_weights:      
            print('=> add layer-wise noise')
            #model_noisy=add_gaussian_noise_layerwise(model, noise_config, device)
        # do it anyway                 
        model_noisy, delta_w = weights_to_noisy_weights(model, temperature_weights, drift_time, row_ou_weights, col_ou_weights)

        #exit()
        if args.task_type=="graph":
            print("Computing hessian score")
            eigenvals, eigenvecs = compute_hessian_eigenthings(
                                    model, # without noise - pre-trained
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
        # get from top in case of cnn or transformer, compute here in case of graphs...
        # reason for this is: eigenthings change in graphs based on mapping of adj matrix, so they need to re-computed in each moo iteration.
        print("Eigenvecs:")
        print(eigenvecs)
        print('eigenvecs shape',eigenvecs.shape)
        print("Eigenvals:")
        print(eigenvals)
        #exit()

        if args.enable_noise_weights: 
            nonidealities=layer_score_func(model_noisy, delta_w, eigenvecs, eigenvals)
        else:
            nonidealities=layer_score_func(model, delta_w, eigenvecs, eigenvals)

        #exit()
    else:
        print('model not in this path..check..')
        exit()

    return nonidealities

def layer_score_func(model, delta_w, eigenvecs, eigenvals):
    # use these to find layer parameter importance
    layer_score=[]
    s_vec_list=[]
    num_params_list=[]
    with torch.no_grad():
        idx=0
        layer_idx=0
        for name, p in model.named_parameters():
            if args.task_type=="graph": # works for GCN
                if 'weight' in name and 'layer_norm' not in name:
                    weights = p
                    num_params=weights.size()[0]*weights.size()[1]
                    num_params_list.append(num_params)
                    start=idx
                    end=idx+num_params
                    if args.enable_noise_weights: 
                        w1=delta_w[layer_idx].detach().cpu().numpy()
                    else:
                        w1=weights.detach().cpu().numpy()
                    w1=w1.flatten()
                    s=0
                    for i in range(args.num_eigenthings):
                        q=eigenvecs[i][start:end]
                        s+=abs(eigenvals[i])*q*q
                    s=s*w1*w1
                    idx=idx+num_params
                    layer_idx+=1
                    s_vec_list.append(s) # vectors
                    layer_score.append(sum(s))#/num_params)

            elif args.task_type=="image":
                if 'conv' in name and 'weight' in name and 'norm' not in name and 'lin_r' not in name or 'linear' in name and 'bias' not in name:
                    weights = p # weights of kth layer
                    num_params=1
                    for i in range(len(weights.size())):
                        num_params=num_params*weights.size()[i]
                    num_params_list.append(num_params)
                    start=idx
                    end=idx+num_params
                    if args.enable_noise_weights: 
                        w1=delta_w[layer_idx].detach().cpu().numpy()
                    else:
                        w1=weights.detach().cpu().numpy()
                    w1=w1.flatten()
                    s=0
                    for i in range(args.num_eigenthings):
                        q=eigenvecs[i][start:end]
                        s+=abs(eigenvals[i])*q*q
                    s=s*w1*w1
                    idx=idx+num_params
                    layer_idx+=1
                    s_vec_list.append(s) # vectors
                    layer_score.append(sum(s))#/num_params)
                    
        print('layer_score', layer_score)
        print('sum of all hessians (full N/W)',sum(layer_score) )

        max_value = max(layer_score)

        if max_value==0:
            print('check hessian')
            exit()

        norm_score=[x / max_value for x in layer_score]
        print('norm_layer_score', norm_score)
        
        if args.enable_noise_weights==0: # for gnn dont want to plot because adj noise changes the eigenthings
            plot_list(norm_score, 'hessian_score') # PLOT hessian score
            #exit()
        if args.task_type=="image": # can plot for cnn
            plot_list(norm_score, 'hessian_score')

        '''
        #print('s_vec_list', s_vec_list) # list of vectors of unequal lengths
        s_arr=np.concatenate(np.array(s_vec_list, dtype=object)) 
        #print(np.shape(s_arr))
        #exit()
        # find top k% scores and to which bin they belong
        top_k_list=[1,3,5]
        for k_percent in top_k_list:
            top_k(s_arr, num_params_list, k_percent)
            print("\n")
        '''
    return sum(layer_score)


def weights_to_noisy_weights(model, temperature, time, row_ou, col_ou, alpha=1, temp_ref=0):
    """
    Convert weights to conductance values.
    Args:
        weights (torch.Tensor): The weight values from the GNN layers.
    Returns:
        torch.Tensor: noisy weights values.
    """
    # Map weights from [-1, 1] to [min_g, max_g]
    #kb=8.6*(10**-5)
    #Ea=1.2
    freq = 100  # operating frequency in MHz
    kb = 1.38e-23  # Boltzmann const
    q = 1.6e-19  # electron charge
    input_reduced=0.01
    err_constant=1
    R_wire=1

    IR=R_wire*err_constant*np.sqrt(row_ou**2+col_ou**2) # array
    print('IR', IR)
    
    g_min_ni = args.g_min * (1 + alpha * (temperature - temp_ref)) # temp- tensor of length arg.layers
    #g_min_ni=args.g_min * (math.e**(-(Ea/(kb*temperature))))
    g_drift = args.g_max * (time)**(-args.drift_coeff) # scalar
    print('g_drift', g_drift)
    g_max_ni =  1/((1/g_drift)+IR)  # array
    delta_g=(g_max_ni - g_min_ni) / (2 ** 7)
    print('delta_g',delta_g) #tensor of length arg.layers

    max_val = 2**args.num_bits - 1
    min_val = 0

    print('new_gmin', g_min_ni)
    print('new_gmax',g_max_ni)

    if g_max_ni[0] < g_min_ni[0]:
        print('g_max_ni < g_min_ni... check this first')
        exit()
    
    #exit()
    layer_count=0
    delta_w_list=[]
    with torch.no_grad():
        for name, p in model.named_parameters():
            if args.task_type=="image":
                #if 'conv' in name and 'weight' in name and 'layer_norm' not in name and 'lin_r' not in name:
                if 'conv' in name and 'weight' in name and 'norm' not in name and 'lin_r' not in name or 'linear' in name and 'bias' not in name:
                    #print(name)
                    weights = p # weights of kth layer  
                    #print('Pre-trained weights',weights)              
                    conductance = g_min_ni[layer_count] + (g_max_ni[layer_count] - g_min_ni[layer_count]) * (weights + 1) / 2
                    #print('W to conductance',conductance)
                
                    grms = torch.sqrt(
                    conductance * freq * (4 * kb * temperature[layer_count] + 2 * q * input_reduced) / (input_reduced ** 2) \
                    + (delta_g[layer_count] / 3) ** 2)
                    grms[torch.isnan(grms)] = 0
                    grms[grms.eq(float('inf'))] = 0
                    #print('grms',grms)
                    #rand_g = torch.Tensor(conductance.shape).cuda()
                    #rand_g.normal_(0, 1)
                    #print('rand_g', rand_g)
                    g_noisy= conductance+grms#+rand_g
                    #print('conductance after adding rms noise',conductance)
                    #quantized_conductance = torch.round(conductance * max_val).clamp(min_val, max_val).int()
                    quantized_conductance = torch.round(g_noisy/delta_g[layer_count])*delta_g[layer_count]
                    #print('quantized_conductance',quantized_conductance)
                    noisy_weights = (quantized_conductance.float() - g_min_ni[layer_count]) / (g_max_ni[layer_count] - g_min_ni[layer_count]) * 2 - 1
                    #noisy_weights = ((conductance.float() - g_min_ni[layer_count]) / (g_max_ni - g_min_ni[layer_count])) * 2 - 1
                    #print('Noisy w:',noisy_weights)
                    delta_w=noisy_weights-weights
                    #print('delta_w',delta_w)
                    delta_w_list.append(delta_w)
                    #print('delta_w mean', torch.mean(delta_w))
                    p.add_(delta_w)
                    layer_count+=1
                    #exit()
                    #print('noisy model params',p) # should be same as noisy_weights
            elif args.task_type=="graph":
                if 'weight' in name and 'layer_norm' not in name:
                    weights = p # weights of kth layer  
                    #print('Pre-trained weights',weights)              
                    conductance = g_min_ni[layer_count] + (g_max_ni[layer_count] - g_min_ni[layer_count]) * (weights + 1) / 2
                    #print('W to conductance',conductance)
                
                    grms = torch.sqrt(
                    conductance * freq * (4 * kb * temperature[layer_count] + 2 * q * input_reduced) / (input_reduced ** 2) \
                    + (delta_g[layer_count] / 3) ** 2)
                    grms[torch.isnan(grms)] = 0
                    grms[grms.eq(float('inf'))] = 0
                
                    g_noisy= conductance+grms#+rand_g
                    quantized_conductance = torch.round(g_noisy/delta_g[layer_count])*delta_g[layer_count]
                    noisy_weights = (quantized_conductance.float() - g_min_ni[layer_count]) / (g_max_ni[layer_count] - g_min_ni[layer_count]) * 2 - 1
                    delta_w=noisy_weights-weights
                    delta_w_list.append(abs(delta_w)) # updated to abs()
                    p.add_(delta_w)
                    layer_count+=1
                
    return model, delta_w_list

def add_noise_adj(adj, time, temperature, device, row_ou, col_ou): # convert to dense, add noise, convert back to sparse
    print('inside add_noise_adj func')
    if args.arch=="GCN":
        adj=adj.to_dense().detach().cpu()#.to(device) 
        #print('edge index format',adj)
    else:
        adj=to_dense_adj(adj, max_num_nodes=4) # pytroch_geometric function #
    #print('=> dense adj:', adj)
    #print('=> adj size:', adj.size())

    adj=add_noise_chunked(adj, time, temperature, device, row_ou, col_ou) # [crossbar_size/(16-bit float/2-bit/cell)]
    #print('=> dense adj with noise:', adj)
    #exit()

    # plot adj matrix
    #plot_dist(adj, 'adjacency_matrix', 'post')
    
    if not args.arch=="GCN": # GAT or SAGE
        adj,adj_attr=dense_to_sparse(adj) # pytroch_geometric function

    return adj

def add_noise_chunked(tensor, time, temperature, device, row_ou, col_ou, chunk_size=int(128/8), temp_ref=0, alpha=1):
    freq = 100  # operating frequency in MHz
    kb = 1.38e-23  # Boltzmann const
    q = 1.6e-19  # electron charge
    input_reduced=0.01

    err_constant=10
    R_wire=1

    IR=R_wire*err_constant*np.sqrt(row_ou**2+col_ou**2) # scalar
    
    g_min_ni=args.g_min * (1 + alpha * (temperature - temp_ref)) # 
    #g_min_ni=args.g_min * (math.e**(-(Ea/(kb*temperature))))
    #g_max_ni=args.g_max * (time)**(-args.drift_coeff) # scalar
    g_drift = args.g_max * (time)**(-args.drift_coeff) # scalar
    g_max_ni =  1/((1/g_drift)+IR)  # scalar
    delta_g=(g_max_ni - g_min_ni) / (2 ** 7)
    #print('delta_g',delta_g) #tensor of length arg.layers

    #g_min_ni=torch.tensor(g_min_ni).to(device)
    #g_max_ni=torch.tensor(g_max_ni).to(device)
    #delta_g=torch.tensor(delta_g).to(device)

    max_val = 2**args.num_bits - 1
    min_val = 0

    print('new_gmin', g_min_ni)
    print('new_gmax',g_max_ni)
    
    if g_max_ni < g_min_ni:
        print('g_max_ni < g_min_ni... check this first')
        exit()
    
    for i in range(0, tensor.size(0), chunk_size):
        for j in range(0, tensor.size(0), chunk_size):
            #print(i)
            chunk = tensor[i:i + chunk_size, j:j + chunk_size]
            #print('chunk size', chunk.size())
            if torch.any(chunk != 0):
                conductance = g_min_ni + (g_max_ni - g_min_ni) * (chunk)
                grms = torch.sqrt(
                    conductance * freq * (4 * kb * temperature + 2 * q * input_reduced) / (input_reduced ** 2) \
                    + (delta_g / 3) ** 2)
                grms[torch.isnan(grms)] = 0
                grms[grms.eq(float('inf'))] = 0
                #print('grms',grms)
                g_noisy= conductance+grms
                quantized_conductance = torch.round(g_noisy/delta_g)*delta_g
                
                noisy_chunk = (quantized_conductance.float() - g_min_ni) / (g_max_ni - g_min_ni) #* 2 - 1
                
                #print('Noisy adj:',noisy_chunk)
                tensor[i:i + chunk_size, j:j + chunk_size] = noisy_chunk

                #noise = torch.randn_like(chunk) * args.adj_std + mean
                #tensor[i:i + chunk_size, j:j + chunk_size] = chunk + noise # add noise to diag blocks
            else:
                tensor[i:i + chunk_size, j:j + chunk_size] = chunk # no noise on non-diagonal elements
            #exit()
    print('% Number of zero elements',torch.sum(tensor == 0).item()/(tensor.size(0)*tensor.size(1))*100)
    return tensor
'''    
def add_gaussian_noise_adj(adj, device):
    adj=adj.to_dense().to(device) # convert to dense, add noise, convert back to sparse
    print('=> dense adj:', adj)
    print('=> adj size:', adj.size())

    adj=add_noise_chunked(adj, mean=0, chunk_size=int(128/8)) # [crossbar_size/(16-bit float/2-bit/cell)]
    print('=> dense adj with noise:', adj)

    # plot adj matrix
    plot_dist(adj, 'adjacency_matrix', 'post')

    #adj_noise = torch.randn(adj.size(), dtype=torch.float16, device=device) * adj_std_dev
    #print('=> adj_noise:', adj_noise)
    #adj=adj+adj_noise
    #adj=adj.to_sparse().to(device) # with noise added, adj is no more sparse, so i commented this
    return adj

def add_noise_chunked(tensor, mean, chunk_size):
    """Adds Gaussian noise to a large tensor in chunks."""
    #noisy_tensor = torch.empty_like(tensor)
    for i in range(0, tensor.size(0), chunk_size):
        for j in range(0, tensor.size(0), chunk_size):
            #print(i)
            chunk = tensor[i:i + chunk_size, j:j + chunk_size]
            #print('chunk size', chunk.size())
            if torch.any(chunk != 0):
                noise = torch.randn_like(chunk) * args.adj_std + mean
                tensor[i:i + chunk_size, j:j + chunk_size] = chunk + noise # add noise to diag blocks
            else:
                tensor[i:i + chunk_size, j:j + chunk_size] = chunk # no noise on non-diagonal elements
            #exit()
            print('% Number of zero elements',torch.sum(tensor == 0).item()/(tensor.size(0)*tensor.size(1)))
    return tensor        
'''
# Function to add Gaussian noise to model parameters 
# NOT USED
'''
def add_gaussian_noise(model, mean=0.0, std=0.01):
    with torch.no_grad():  # Ensure no gradients are computed
        for param in model.parameters():
            noise = torch.normal(mean=mean, std=std, size=param.size())
            param.add_(noise)
'''        

def add_gaussian_noise_layerwise(model, noise_config, device):
    """
    Adds Gaussian noise to specific layers of the model based on the noise configuration.

    """
    with torch.no_grad():
         
        for name, layer in model.named_parameters():
            #print('inside for_1')
            #print('name',name)
            #print('layer',layer)  
            if name in noise_config:
                #print('inside if')  
                mean, std = noise_config[name]

                noise = torch.normal(mean=mean, std=std, size=layer.size()).to(device)
                plot_dist(layer, name, 'pre')
                layer.add_(noise)
                plot_dist(layer, name, 'post')                    
    return model


def plot_dist(input, name, pre_or_post):
    # Convert the tensor to a NumPy array 
    z_np = input.cpu().numpy()
    plt.figure()
    plt.xlim(-1, 1)
    # Plot the histogram
    plt.hist(z_np.flatten(), bins=500)
    plt.title("Gaussian Noise Distribution")
    plt.xlabel("Weights Std Deviation")
    plt.ylabel("Density")
    plt.savefig('saves/'+args.dataset+'_'+args.arch+'/'+pre_or_post+'.'+name+'.png')
    plt.close()

def plot_list(values, filename):
    """
    Plots the given list values with indices on the x-axis and the list values on the y-axis.

    Parameters:
        values (list of float/int): The list of values to plot.
    """
    if not values:
        print("The list is empty. Nothing to plot.")
        return

    ## Generate x-axis indices
    #x = list(range(len(values)))
    # Generate x-axis indices starting from 1
    x = list(range(1, len(values) + 1))

    # Plot the values
    plt.figure(figsize=(8, 4))  # Set the figure size
    plt.plot(x, values, marker='o', linestyle='-', color='b')#, label='Values')  # Line plot
    #plt.title("List Values Plot")
    plt.xlabel("Neural Layer")
    plt.ylabel("Norm Hessian Score")
    plt.xticks(x)  # Ensure x-axis ticks are integers
    plt.grid(True)  # Add a grid for better visualization
    #plt.legend()
    plt.savefig('output/'+args.dataset+'/'+args.arch+'/'+filename+'.png')
    plt.close()

    return

def top_k(s_arr, num_params, k_percent):
    #k_percent = 10  # Change to your desired percentage
    k_values_count = int(len(s_arr) * (k_percent / 100))

    # Step 1: Find indices of top k% values in s_arr
    top_k_indices = np.argsort(s_arr)[-k_values_count:]

    # Step 2: Assign bins to indices
    bin_labels = []
    start_idx = 0

    for i, size in enumerate(num_params):
        bin_labels.extend([i] * size)
        start_idx += size

    bin_labels = np.array(bin_labels[:len(s_arr)])  # In case s_arr size < sum(num_params)

    # Step 3: Count top k% values in each bin
    bin_counts = np.zeros(len(num_params), dtype=int)

    for idx in top_k_indices:
        bin_counts[bin_labels[idx]] += 1

    # Print results
    for i, count in enumerate(bin_counts):
        print(f"Bin {i + 1} has {count:,} values in the top {k_percent}% of s_arr.")
    
    return
