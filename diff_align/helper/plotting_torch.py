import numpy as np
import matplotlib.pyplot as plt
import torch
import seaborn as sns
sns.set_theme(color_codes=True)
from helper.UCR_loader import processed_UCR_data, load_txt_file, np_to_dataloader
from tslearn.datasets import UCR_UEA_datasets
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import rbf_kernel, pairwise_kernels
import scipy
from collections import Counter
from models.train_model import test


def plot_mean_signal(X_aligned_within_class, X_within_class, ratio, class_num, dataset_name, N=10):

    #check data dim
    if len(X_aligned_within_class.shape) < 3:
        X_aligned_within_class = np.expand_dims(X_aligned_within_class, axis=-1)

    # data dims: (number of samples, dim, channels)
    n_signals = len(X_within_class)  # number of samples within class

    # Sample random signals
    input_shape = X_within_class.shape[1:]  # (channels, dims) PyTorch
    signal_len = input_shape[1]
    n_channels = input_shape[0]

    np.random.seed(2021)
    # indices = np.random.choice(n_signals, N)  # N samples
    # X_within_class = X_within_class[indices, :, :]  # get N samples, all channels
    # X_aligned_within_class = X_aligned_within_class[indices, :, :]

    X_within_class = X_within_class[:10, :, :]  # get N samples, all channels
    X_aligned_within_class = X_aligned_within_class[:10, :, :]

    # Compute mean signal and variance
    X_mean_t = np.mean(X_aligned_within_class, axis=0)
    X_std_t = np.std(X_aligned_within_class, axis=0)
    upper_t = X_mean_t + X_std_t
    lower_t = X_mean_t - X_std_t

    X_mean = np.mean(X_within_class, axis=0)
    X_std = np.std(X_within_class, axis=0)
    upper = X_mean + X_std
    lower = X_mean - X_std

    # set figure size
    # [w, h] = ratio  # width, height
    # f = plt.figure(1)
    # plt.style.use('seaborn-darkgrid')
    # f.set_size_inches(w, h)
    # f.set_size_inches(w, n_channels * h)

    title_font = 18
    rows = 2
    cols = 2
    # plot each channel
    for channel in range(n_channels):
        [w, h] = ratio  # width, height
        f = plt.figure(1)
        plt.style.use('seaborn-darkgrid')
        f.set_size_inches(w, h)
        plot_idx = 1
        t = range(input_shape[1])
        # Misaligned Signals
        # if channel == 0:
            # ax1 = f.add_subplot(rows, cols, plot_idx)
        ax1 = f.add_subplot(rows, cols, plot_idx)
        ax1.plot(X_within_class[:, channel,:].T)
        plt.tight_layout()
        plt.xlim(0, signal_len)

        if n_channels == 1:
            #plt.title("%d random test samples" % (N))
            plt.title("Misaligned signals", fontsize=title_font)
        else:
            plt.title("Channel: %d, %d random test samples" % (channel, N))
        plot_idx += 1

        # Misaligned Mean
        # if channel == 0:
        #     ax2 = f.add_subplot(rows, cols, plot_idx)
        ax2 = f.add_subplot(rows, cols, plot_idx)
        if n_channels == 1:
            ax2.plot(t, X_mean[channel], 'r',label=f'Average signal-channel:{channel}')
            ax2.fill_between(t, upper[channel], lower[channel], color='r', alpha=0.2, label=r"$\pm\sigma$")
        else:
            ax2.plot(t, X_mean[channel,:], label=f'Average signal-channel:{channel}')
            ax2.fill_between(t, upper[channel], lower[channel], color='r', alpha=0.2, label=r"$\pm\sigma$")

        plt.legend(loc='upper right', fontsize=12, frameon=True)
        plt.xlim(0, signal_len)

        if n_channels ==1:
            plt.title("Misaligned average signal", fontsize=title_font)
        else:
            plt.title(f"Channel: {channel}, Test data mean signal ({N} samples)")
            ax2.fill_between(t, upper[channel], lower[channel], color='r', alpha=0.2, label=r"$\pm\sigma$")

        plot_idx += 1


        # Aligned signals
        # if channel == 0:
        #     ax3 = f.add_subplot(rows, cols, plot_idx)
        ax3 = f.add_subplot(rows, cols, plot_idx)
        ax3.plot(X_aligned_within_class[:, channel,:].T)
        plt.title("DTAN aligned signals", fontsize=title_font)
        plt.xlim(0, signal_len)

        plot_idx += 1

        # Aligned Mean
        # if channel == 0:
        #     ax4 = f.add_subplot(rows, cols, plot_idx)
        ax4 = f.add_subplot(rows, cols, plot_idx)
        # plot transformed signal
        ax4.plot(t, X_mean_t[channel,:], label=f'Average signal-channel:{channel}')
        if n_channels == 1:
            ax4.fill_between(t, upper_t[channel], lower_t[channel], color='#539caf', alpha=0.6, label=r"$\pm\sigma$")
        else:
            ax4.fill_between(t, upper_t[channel], lower_t[channel], color='#539caf', alpha=0.6, label=r"$\pm\sigma$")

        plt.legend(loc='upper right', fontsize=12, frameon=True)
        plt.title("DTAN average signal", fontsize=title_font)
        plt.xlim(0, signal_len)
        plt.tight_layout()
        plot_idx += 1

        plt.savefig(f'{int(class_num)}_{dataset_name}.pdf', format='pdf')

        # plt.suptitle(f"{dataset_name}: class-{class_num}", fontsize=title_font+2)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()
    plt.suptitle(f"{dataset_name}: class-{class_num}", fontsize=title_font+2)


def construct_data(data):
    res = []

    for feature in data.columns:
            res.append(data.loc[:, feature].values.tolist())
  
    return res


def plot_signals(model, device, datadir, dataset_name, train_dataloader, test_dataloader, data_numpy, batch_size):
    # Close any remaining plots
    plt.close('all')

    with torch.no_grad():
        # # Torch channels first
        # # if (datadir):
        # #   X_train, X_test, y_train, y_test = load_txt_file(datadir, dataset_name)
        # # else:
        # #   X_train, y_train, X_test, y_test = UCR_UEA_datasets().load_dataset(dataset_name)
        # # X_train, X_test, y_train, y_test = processed_UCR_data(X_train, X_test, y_train, y_test)

        # fdir = os.path.join(datadir, dataset_name)
        # assert os.path.isdir(fdir), f"{fdir}. {dataset_name} could not be found in {datadir}"
        # # again, for file names
        # f_name = os.path.join(fdir, dataset_name)
        # # lag=190 ##ecg
        # lag=60
        # # df= pd.read_csv(f_name+'.csv', header=0) ##ecg
        # # df= pd.read_csv(f_name+'.csv',header=None) ##art
        # df1= pd.read_csv(f_name+'_train.csv',index_col=0)
        # df2= pd.read_csv(f_name+'_test.csv',index_col=0)

        # if 'attack' in df1.columns:
        #     data_Y_train=df1['attack']

        #     df1 = df1.drop(columns=['attack'])
            
        # if 'attack' in df2.columns:
        #     data_Y_test=df2['attack']

        #     df2 = df2.drop(columns=['attack'])
        # data_X_train=construct_data(df1)
        # data_X_test=construct_data(df2)
        # # data1=df1.values
        # # data2=df2.values
        # # # data_X=data[:,0] ## art
        # # data_X_train=data1
        # # data_X_test=data2[:,:-1]
        # # data_Y_test=data2[:,-1]
        # # data_X=np.array(data[:,1], dtype=np.float32) ##ecg
        # # data_seq=np.empty((0,lag))
        # # data_seq_train=np.empty((0,lag,data_X_train.shape[1]))
        # # data_seq_test=np.empty((0,lag,data_X_test.shape[1]))
        # # data_seq=np.empty((0,lag,3))
        # data_lab=[]
        # data_lab_train=[]
        # data_lab_test=[]
        # data_lab_org=[]
        # data_seq_train=[]
        # data_seq_test=[]


        # data_X_train = np.array(data_X_train)
        # data_X_test = np.array(data_X_test)

        # node_num, train_len = np.shape(data_X_train)
        # node_num, test_len =  np.shape(data_X_test)


        # for i in range(lag,train_len):
        #         ft = data_X_train[:,i-lag:i]
        #         data_seq_train.append(ft)
        #         data_lab_train.append(1)
        # data_seq_train = np.stack(data_seq_train)
        # data_lab_train = np.stack(data_lab_train)
        # # data_seq_train = torch.stack(data_seq_train).contiguous()
        # # data_lab_train = torch.stack(data_lab_train).contiguous()

        # for i in range(lag,test_len):
        #         ft = data_X_test[:,i-lag:i]
        #         data_seq_test.append(ft)
        #         data_lab_test.append(1)
        # data_seq_test = np.stack(data_seq_test)
        # data_lab_test = np.stack(data_lab_test)

        # # for i in range(0,len(data)-lag-2,30):

        # # for i in range(0,len(data_X_train)-lag):
        # #     data_seq_train=np.append(data_seq_train,[np.array(data_X_train[i:i+lag])],axis=0)
        # #     # data_seq=np.append(data_seq,[np.array(data_X[i+1:i+lag+1])],axis=0)
        # #     # data_seq=np.append(data_seq,[np.array(data_X[i+2:i+lag+2])],axis=0)
        # #     # data_lab_org.append(np.int(np.any(data_Y[i:i+lag]<0)))
        # #     # data_lab_org.append(np.int(np.any(data_Y[i+1:i+lag+1]<0)))
        # #     # data_lab_org.append(np.int(np.any(data_Y[i+2:i+lag+2]<0)))
        # #     # data_lab.append([1,1,1])
        # #     # data_lab_train.append(Counter(data_Y_train[i:i+lag]).most_common(1)[0][0])
        # #     data_lab_train.append(1)
            
        # # for i in range(0,len(data_X_test)-lag):
        # #     data_seq_test=np.append(data_seq_test,[np.array(data_X_test[i:i+lag])],axis=0)
        # #     data_lab_test.append(Counter(data_Y_test[i:i+lag]).most_common(1)[0][0])

        # # X_train, X_test, y_train, y_test = train_test_split(data_seq, data_lab_org, test_size=0.2, random_state=42)
        # # X_train, X_test, y_train, y_test = processed_UCR_data(X_train, X_test, y_train, y_test)
        # X_train, X_test, y_train, y_test = processed_UCR_data(data_seq_train, data_seq_test, data_lab_train, data_lab_test)

        # data =[X_train, X_test]
        # labels_org = [y_train, y_test]
        # labels = [np.zeros(len(y_train)), np.zeros(len(y_test))]

        # train_dataloader = np_to_dataloader(X_train, y_train, batch_size, shuffle=False)
        transformed_data_numpy_train = test(train_dataloader, device, model)
        print(transformed_data_numpy_train.shape)
        np.save('/home/abilasha/Downloads/mts_ad/dtan-master/examples/data/{0}/result_train.npy'.format(dataset_name), transformed_data_numpy_train)
        # test_dataloader = np_to_dataloader(X_test, y_test, batch_size, shuffle=False)
        transformed_data_numpy_test = test(test_dataloader, device, model)
        print(transformed_data_numpy_test.shape)
        np.save('/home/abilasha/Downloads/mts_ad/dtan-master/examples/data/{0}/result_test.npy'.format(dataset_name), transformed_data_numpy_test)

        sns.set_style("whitegrid")
            
        X_within_class = data_numpy

        X_aligned_within_class = transformed_data_numpy_test

        plot_mean_signal(X_aligned_within_class, X_within_class, ratio=[10,4],
                                 class_num=0, dataset_name=f"{dataset_name}")

        # set_names = ["train", "test"]
        # for i in range(2):
        #     # torch dim
        #     X = torch.Tensor(data[i]).to(device)
        #     # y = labels[i]
        #     y = labels_org[i]
        #     classes = np.unique(y)
        #     transformed_input_tensor, thetas = model(X, return_theta=True)

        #     data_numpy = X.data.cpu().numpy()
        #     transformed_data_numpy = transformed_input_tensor.data.cpu().numpy()

        #     sns.set_style("whitegrid")

        #     #fig, axes = plt.subplots(1,2)
        #     # print(classes)
        #     for label in classes:
        #         # print(label)
        #         class_idx = y == label
        #         X_within_class = data_numpy[class_idx]
        #         X_aligned_within_class = transformed_data_numpy[class_idx]
        #         # np.save('/home/abilasha/Downloads/mts_ad/dtan-master/examples/data/{0}/result_{1}.npy'.format(dataset_name,set_names[i]), X_aligned_within_class)
        #         print(X_aligned_within_class.shape, X_within_class.shape)
                
        #         plot_mean_signal(X_aligned_within_class, X_within_class, ratio=[10,4],
        #                          class_num=label, dataset_name=f"{dataset_name}-{set_names[i]}")

        #         # for j in range(data_numpy.shape[1]):
        #         #     per_channel_loss = 0 
        #         #     X_weight_mean=np.empty((0,60))
        #         #     dist_mat=scipy.spatial.distance.cdist(X_aligned_within_class[:,j,:],X_aligned_within_class[:,j,:])+1e-8
        #         #     width=np.median(dist_mat)
        #         #     covar = rbf_kernel(X_aligned_within_class[:,j,:],gamma=1/(2*width*width))
                    
        #         #     plt.imshow(covar, cmap='hot', interpolation='nearest', origin = 'lower', extent = (0, covar.shape[1]-0, 0, covar.shape[0]-0))
        #         #     plt.colorbar()
        #         #     plt.clim(0, 1)
        #         #     plt.show()
        #         #     g = sns.clustermap(covar, figsize=(5,5))
        #         #     plt.show()
