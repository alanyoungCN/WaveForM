import os
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag, size,
                data_path,
                scale=True):
        self.seq_len = size[0]
        self.pred_len = size[1]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.scale = scale

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        cols = list(df_raw.columns)
        cols.remove('OT')
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + ['OT']]
        # print(cols)
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [
            0,  # train
            num_train - self.seq_len,  # validation
            len(df_raw) - num_test - self.seq_len  # test
        ]
        border2s = [
            num_train,  # train
            num_train + num_vali,  # validation
            len(df_raw)  # test
        ]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        cols_data = df_raw.columns[1:]
        df_data = df_raw[cols_data]

        
        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]

        return seq_x, seq_y

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
