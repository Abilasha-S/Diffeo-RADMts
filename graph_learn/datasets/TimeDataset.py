import torch
from torch.utils.data import Dataset, DataLoader

import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np


class TimeDataset(Dataset):
    # def __init__(self, raw_data, edge_index, mode='train', config = None):
    def __init__(self, raw_data, emb_data, edge_index, mode='train', config = None):
        self.raw_data = raw_data
        self.emb_data = emb_data                        ##diff
        self.config = config
        self.edge_index = edge_index
        self.mode = mode

        x_data = raw_data[:-1]
        labels = raw_data[-1]

        data = x_data

        # to tensor
        data = torch.tensor(data).double()
        labels = torch.tensor(labels).double()

        # self.x, self.y, self.labels_recons , self.labels_pred = self.process(data, labels)                        ##for diff comment
        self.x, self.y, self.labels_recons , self.labels_pred = self.process(data, self.emb_data, labels)                        ##diff
    
    def __len__(self):
        return len(self.x)


    # def process(self, data, labels):
    def process(self, data, emb_data, labels):
        x_arr, y_arr = [], []
        labels_arr_recons = []
        labels_arr_pred = []

        slide_win, slide_stride = [self.config[k] for k
            in ['slide_win', 'slide_stride']
        ]
        is_train = self.mode == 'train'

        node_num, total_time_len = data.shape

        rang = range(slide_win, total_time_len, slide_stride) if is_train else range(slide_win, total_time_len)
        
        for i in rang:

            # ft = data[:, i-slide_win:i]                        ##for diff comment
            tar = data[:, i]###Pred

            # x_arr.append(ft)                        ##for diff comment
            y_arr.append(tar)###Pred
            # labels_arr_recons.append(torch.median(labels[i-slide_win:i]))
            labels_arr_recons.append(torch.max(labels[i-slide_win:i]))
            labels_arr_pred.append(labels[i])##Pred

        # x = torch.stack(x_arr).contiguous()                        ##for diff comment
        x = torch.Tensor(emb_data)                        ##diff
        # y = x
        labels_recons = torch.Tensor(labels_arr_recons).contiguous()
        y = torch.stack(y_arr).contiguous()##Pred
        labels_pred = torch.Tensor(labels_arr_pred).contiguous()

        return x, y, labels_recons, labels_pred

    def __getitem__(self, idx):

        feature = self.x[idx].double()
        y = self.y[idx].double()

        edge_index = self.edge_index.long()

        label_recons = self.labels_recons[idx].double()
        label_pred = self.labels_pred[idx].double()

        return feature, y, label_recons, label_pred, edge_index





