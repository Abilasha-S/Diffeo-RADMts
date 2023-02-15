from DTAN.smoothness_prior import smoothness_norm
import torch
import gpytorch
from sklearn.metrics.pairwise import rbf_kernel, pairwise_kernels
import numpy as np
import scipy
import math
import matplotlib.pyplot as plt

def alignment_loss(X_trasformed, labels, thetas, n_channels, DTANargs):
    '''
    Torch data format is  [N, C, W] W=timesteps
    Args:
        X_trasformed:
        labels:
        thetas:
        DTANargs:

    Returns:

    '''
    loss = 0
    align_loss = 0
    prior_loss = 0
    n_classes = labels.unique()
    for i in n_classes:
        X_within_class = X_trasformed[labels==i]
        X_weight_mean=np.empty((0,60))
        # X_weight_mean=torch.empty(len(X_within_class),90).cuda()
        inter_loss=0
        if n_channels == 1:
            # Single channel variance across samples

            X_within_class_np=X_within_class.detach().cpu().numpy().squeeze() 
            dist_mat=scipy.spatial.distance.cdist(X_within_class_np,X_within_class_np)
            width=np.median(dist_mat)
            covar = rbf_kernel(X_within_class_np,gamma=1/(2*width*width))
            # covar = rbf_kernel(X_within_class_np,gamma=1/(2*100))
            covar[covar<0.4]=0
            for i in range(covar.shape[1]):
                X_weight_mean=np.append(X_weight_mean,np.array([np.matmul(covar[i],X_within_class.detach().cpu().numpy().squeeze(axis=1))])/np.sum(covar[i]),axis=0)
            loss+=torch.linalg.norm((X_within_class.squeeze(axis=1)-torch.Tensor(X_weight_mean).cuda()), ord=2, dim=1).mean() 

            # for i in range(X_within_class.shape[0]):
            #     # print("vi-vj2",-torch.diagonal(torch.mm((X_within_class.squeeze(axis=1)-X_within_class.squeeze(axis=1)[i]),(X_within_class.squeeze(axis=1)-X_within_class.squeeze(axis=1)[i]).t()),0))
            #     X_weight_mean_inter = torch.div(torch.exp(-torch.norm(X_within_class.squeeze(axis=1)-X_within_class.squeeze(axis=1)[i], dim=1)),2*0.005)
            #     # X_weight_mean_inter = torch.div(torch.exp(-torch.diagonal(torch.mm((X_within_class.squeeze(axis=1)-X_within_class.squeeze(axis=1)[i]),(X_within_class.squeeze(axis=1)-X_within_class.squeeze(axis=1)[i]).t()),0)),2*1)
            #     # print("sim of j with vi",X_weight_mean_inter.shape)
            #     # print("weight sum",torch.sum(X_weight_mean_inter))
            #     # print("weighted mean",torch.matmul(X_weight_mean_inter,X_within_class.squeeze(axis=1)))
            #     # inter_loss += torch.linalg.norm((X_within_class.squeeze(axis=1)[i]-torch.div(torch.matmul(X_weight_mean_inter,X_within_class.squeeze(axis=1)),torch.sum(X_weight_mean_inter))), ord=2) 
            #     inter_loss += torch.linalg.norm((X_within_class.squeeze(axis=1)[i]-torch.matmul(X_weight_mean_inter,X_within_class.squeeze(axis=1))), ord=2) 
            #     # print("inter_loss",inter_loss)    
            # loss+=inter_loss/(len(X_within_class))
            # # print("loss",loss)

            # X_mean=X_within_class.mean(dim=0)
            # loss+=torch.linalg.norm((X_within_class-X_mean), ord=1, dim=0).mean()
            # loss += X_within_class.var(dim=0, unbiased=False).mean()
        else:
            # variance between signals in each channel (dim=1)
            # mean variance of all channels and samples (dim=0)

            # X_within_class_np=X_within_class.detach().cpu().numpy()

            # for j in range(n_channels):
            #     per_channel_loss = 0 
            #     X_weight_mean=np.empty((0,60))
            #     dist_mat=scipy.spatial.distance.cdist(X_within_class_np[:,j,:],X_within_class_np[:,j,:])+1e-8
            #     width=np.median(dist_mat)
            #     covar = rbf_kernel(X_within_class_np[:,j,:],gamma=1/(2*width*width))
            #     # covar = rbf_kernel(X_within_class_np,gamma=1/(2*100))
            #     # covar[covar<0.4]=0
            #     for i in range(covar.shape[1]):
            #         X_weight_mean=np.append(X_weight_mean,np.array([np.matmul(covar[i],X_within_class_np[:,j,:])])/np.sum(covar[i]),axis=0)
            #     per_channel_loss+=torch.linalg.norm((X_within_class[:,j,:]-torch.Tensor(X_weight_mean).cuda()), ord=2, dim=1).mean()

            # loss+=per_channel_loss/(n_channels)

            # X_mean=X_within_class.mean(dim=0)
            # per_channel_loss=torch.linalg.norm((X_within_class-X_mean), ord=1, dim=1).mean(dim=0)

            per_channel_loss = X_within_class.var(dim=1, unbiased=False).mean(dim=0)
            per_channel_loss = per_channel_loss.mean()
            loss += per_channel_loss

    loss /= len(n_classes)
    # Note: for multi-channel data, assues same transformation (i.e., theta) for all channels
    if DTANargs.smoothness_prior:
        for theta in thetas:
            # alignment loss takes over variance loss
            # larger penalty when k increases -> coarse to fine
            prior_loss += 0.1*smoothness_norm(DTANargs.T, theta, DTANargs.lambda_smooth, DTANargs.lambda_var, print_info=False)
        loss += prior_loss
    return loss
