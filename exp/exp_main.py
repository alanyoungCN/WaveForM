from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic

from models import WaveForM

from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric

import numpy as np
import torch
import torch.nn as nn
from torch import optim

import os
import time

import warnings
import numpy as np

from utils import color

warnings.filterwarnings('ignore')


class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

    def _build_model(self):
        model_dict = {
            'WaveForM': WaveForM,
        }
        model = model_dict[self.args.model].Model(self.args).float()

        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                outputs = self.model(batch_x)
                outputs = outputs[:, :, :]
                batch_y = batch_y[:, :, :].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            color.cprint(f'begin epoch {epoch+1:>3}/{self.args.train_epochs}', color.OKBLUE, '\n')
            for i, (batch_x, batch_y) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                
                outputs = self.model(batch_x)
                outputs = outputs[:, :, :]
                batch_y = batch_y[:, :, :].to(self.device)
                loss = criterion(outputs, batch_y)
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    speed = (time.time() - time_now) / iter_count
                    print(f'iters:{i+1:>5}/{train_steps} loss: {loss.item():5.3f} speed: {speed:5.3f} s/iter')
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                model_optim.step()

            color.cprint(f'epoch {epoch+1:>3}/{self.args.train_epochs} time: {time.time() - epoch_time:>5.2f} secs', color.OKBLUE, '\n')
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            color.cprint(
                f'After Epoch {epoch+1:>3} Steps: {train_steps:>4} Train Loss: {train_loss:5.3f} Vali Loss: {vali_loss:5.3f} Test Loss: {test_loss:5.3f}',
                color.OKBLUE,
                '\n'
            )
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)


                outputs = self.model(batch_x)

                outputs = outputs[:, :, :]
                batch_y = batch_y[:, :, :].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs
                true = batch_y
                preds.append(pred)
                trues.append(true)

        preds = np.array(preds)
        trues = np.array(trues)
        # print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        # print('test shape:', preds.shape, trues.shape)


        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)

        print('mse:{}, mae:{}, rmse:{}'.format(np.mean(mse), np.mean(mae), np.mean(rmse)))
        f = open("result_average.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, rmse:{}-----------average\n'.format(np.mean(mse), np.mean(mae), np.mean(rmse)))
        f.write('\n')
        f.close()

        f = open("result.txt", 'a')
        f.write('\n')
        f.write(setting + "  \n")
        
        for i in range(self.args.pred_len):
            f.write('mse:{}, mae:{}, rmse:{}'.format(mse[i], mae[i], rmse[i]))
            f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return