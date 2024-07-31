import torch
import torch.nn as nn
import torch.functional as F
import lightning.pytorch as pyl
import numpy as np
import copy
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
from argparse import Namespace
from lightning.pytorch.loggers import WandbLogger

from pretrain_dataset import StockDataset
from data_provider.data_factory import data_provider
from data_provider.m4 import M4Meta

from losses import smape_loss
import TimesNetModel

torch.set_float32_matmul_precision('high')

class TimesNet(pyl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.model = TimesNetModel.Model(configs=config)
        self.args = config
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        (batch_x, batch_y, batch_x_mark, batch_y_mark) = batch
        (batch_x, batch_y, batch_x_mark, batch_y_mark) = (batch_x.type(torch.cuda.FloatTensor), batch_y.type(torch.cuda.FloatTensor), batch_x_mark.type(torch.cuda.FloatTensor), batch_y_mark.type(torch.cuda.FloatTensor))
        dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
        dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

        outputs = self.model(batch_x, None, dec_inp, None)

        f_dim = -1 if self.args.features == 'MS' else 0
        outputs = outputs[:, -self.args.pred_len:, f_dim:]
        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

        batch_y_mark = batch_y_mark[:, -self.args.pred_len:, f_dim:].to(self.device)
        criterion = smape_loss()
        loss_value = criterion(batch_x, self.args.frequency_map, outputs, batch_y, batch_y_mark)
        
        self.log("loss", loss_value)

        return loss_value
    
    def validation_step(self, batch, batch_idx):
        # training_step defines the train loop.
        (batch_x, batch_y, batch_x_mark, batch_y_mark) = batch
        (batch_x, batch_y, batch_x_mark, batch_y_mark) = (batch_x.type(torch.cuda.FloatTensor), batch_y.type(torch.cuda.FloatTensor), batch_x_mark.type(torch.cuda.FloatTensor), batch_y_mark.type(torch.cuda.FloatTensor))
        dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
        dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

        outputs = self.model(batch_x, None, dec_inp, None)

        f_dim = -1 if self.args.features == 'MS' else 0
        outputs = outputs[:, -self.args.pred_len:, f_dim:]
        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

        batch_y_mark = batch_y_mark[:, -self.args.pred_len:, f_dim:].to(self.device)
        criterion = smape_loss()
        loss_value = criterion(batch_x, self.args.frequency_map, outputs, batch_y, batch_y_mark)
        
        self.log("val_loss", loss_value, prog_bar=True)

        return loss_value


    def test_step(self, batch, batch_idx):

        (batch_x, batch_y, batch_x_mark, batch_y_mark) = batch
        (batch_x, batch_y, batch_x_mark, batch_y_mark) = (batch_x.type(torch.cuda.FloatTensor), batch_y.type(torch.cuda.FloatTensor), batch_x_mark.type(torch.cuda.FloatTensor), batch_y_mark.type(torch.cuda.FloatTensor))
        dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
        dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

        outputs = self.model(batch_x, None, dec_inp, None)

        f_dim = -1 if self.args.features == 'MS' else 0
        outputs = outputs[:, -self.args.pred_len:, f_dim:]
        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

        batch_y_mark = batch_y_mark[:, -self.args.pred_len:, f_dim:].to(self.device)
        criterion = smape_loss()
        loss_value = criterion(batch_x, self.args.frequency_map, outputs, batch_y, batch_y_mark)
        
        self.log("val_loss", loss_value, prog_bar=True)

        return loss_value
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: self.args.learning_rate * (0.5 ** ((epoch - 1) // 1)))
        return [optimizer], [lr_scheduler]

def get_data(args, flag):
    data_set, data_loader = data_provider(args, flag)
    return data_set, data_loader    

configs = []

# configs.append({'task_name': 'short_term_forecast', 'is_training': 1, 'model_id': 'm4_Hourly', 'model': 'TimesNet', 'data': 'm4', 'root_path': './dataset/m4', 'data_path': 'ETTh1.csv', 'features': 'M', 'target': 'OT', 'freq': 'h', 'checkpoints': './checkpoints/', 'seq_len': 96, 'label_len': 48, 'pred_len': 96, 'seasonal_patterns': 'Hourly', 'inverse': False, 'mask_rate': 0.25, 'anomaly_ratio': 0.25, 'expand': 2, 'd_conv': 4, 'top_k': 5, 'num_kernels': 6, 'enc_in': 1, 'dec_in': 1, 'c_out': 1, 'd_model': 32, 'n_heads': 8, 'e_layers': 2, 'd_layers': 1, 'd_ff': 32, 'moving_avg': 25, 'factor': 3, 'distil': True, 'dropout': 0.1, 'embed': 'timeF', 'activation': 'gelu', 'output_attention': False, 'channel_independence': 1, 'decomp_method': 'moving_avg', 'use_norm': 1, 'down_sampling_layers': 0, 'down_sampling_window': 1, 'down_sampling_method': None, 'seg_len': 48, 'num_workers': 10, 'itr': 1, 'train_epochs': 10, 'batch_size': 16, 'patience': 3, 'learning_rate': 0.001, 'des': 'Exp', 'loss': 'SMAPE', 'lradj': 'type1', 'use_amp': False, 'use_gpu': True, 'gpu': 0, 'use_multi_gpu': False, 'devices': '0,1,2,3', 'p_hidden_dims': [128, 128], 'p_hidden_layers': 2, 'use_dtw': False, 'augmentation_ratio': 0, 'seed': 2, 'jitter': False, 'scaling': False, 'permutation': False, 'randompermutation': False, 'magwarp': False, 'timewarp': False, 'windowslice': False, 'windowwarp': False, 'rotation': False, 'spawner': False, 'dtwwarp': False, 'shapedtwwarp': False, 'wdba': False, 'discdtw': False, 'discsdtw': False, 'extra_tag': ''})
# configs.append({'task_name': 'short_term_forecast', 'is_training': 1, 'model_id': 'm4_Weekly', 'model': 'TimesNet', 'data': 'm4', 'root_path': './dataset/m4', 'data_path': 'ETTh1.csv', 'features': 'M', 'target': 'OT', 'freq': 'h', 'checkpoints': './checkpoints/', 'seq_len': 96, 'label_len': 48, 'pred_len': 96, 'seasonal_patterns': 'Weekly', 'inverse': False, 'mask_rate': 0.25, 'anomaly_ratio': 0.25, 'expand': 2, 'd_conv': 4, 'top_k': 5, 'num_kernels': 6, 'enc_in': 1, 'dec_in': 1, 'c_out': 1, 'd_model': 32, 'n_heads': 8, 'e_layers': 2, 'd_layers': 1, 'd_ff': 32, 'moving_avg': 25, 'factor': 3, 'distil': True, 'dropout': 0.1, 'embed': 'timeF', 'activation': 'gelu', 'output_attention': False, 'channel_independence': 1, 'decomp_method': 'moving_avg', 'use_norm': 1, 'down_sampling_layers': 0, 'down_sampling_window': 1, 'down_sampling_method': None, 'seg_len': 48, 'num_workers': 10, 'itr': 1, 'train_epochs': 10, 'batch_size': 16, 'patience': 3, 'learning_rate': 0.001, 'des': 'Exp', 'loss': 'SMAPE', 'lradj': 'type1', 'use_amp': False, 'use_gpu': True, 'gpu': 0, 'use_multi_gpu': False, 'devices': '0,1,2,3', 'p_hidden_dims': [128, 128], 'p_hidden_layers': 2, 'use_dtw': False, 'augmentation_ratio': 0, 'seed': 2, 'jitter': False, 'scaling': False, 'permutation': False, 'randompermutation': False, 'magwarp': False, 'timewarp': False, 'windowslice': False, 'windowwarp': False, 'rotation': False, 'spawner': False, 'dtwwarp': False, 'shapedtwwarp': False, 'wdba': False, 'discdtw': False, 'discsdtw': False, 'extra_tag': ''})
# configs.append({'task_name': 'short_term_forecast', 'is_training': 1, 'model_id': 'm4_Daily', 'model': 'TimesNet', 'data': 'm4', 'root_path': './dataset/m4', 'data_path': 'ETTh1.csv', 'features': 'M', 'target': 'OT', 'freq': 'h', 'checkpoints': './checkpoints/', 'seq_len': 96, 'label_len': 48, 'pred_len': 96, 'seasonal_patterns': 'Daily', 'inverse': False, 'mask_rate': 0.25, 'anomaly_ratio': 0.25, 'expand': 2, 'd_conv': 4, 'top_k': 5, 'num_kernels': 6, 'enc_in': 1, 'dec_in': 1, 'c_out': 1, 'd_model': 16, 'n_heads': 8, 'e_layers': 2, 'd_layers': 1, 'd_ff': 16, 'moving_avg': 25, 'factor': 3, 'distil': True, 'dropout': 0.1, 'embed': 'timeF', 'activation': 'gelu', 'output_attention': False, 'channel_independence': 1, 'decomp_method': 'moving_avg', 'use_norm': 1, 'down_sampling_layers': 0, 'down_sampling_window': 1, 'down_sampling_method': None, 'seg_len': 48, 'num_workers': 10, 'itr': 1, 'train_epochs': 10, 'batch_size': 16, 'patience': 3, 'learning_rate': 0.001, 'des': 'Exp', 'loss': 'SMAPE', 'lradj': 'type1', 'use_amp': False, 'use_gpu': True, 'gpu': 0, 'use_multi_gpu': False, 'devices': '0,1,2,3', 'p_hidden_dims': [128, 128], 'p_hidden_layers': 2, 'use_dtw': False, 'augmentation_ratio': 0, 'seed': 2, 'jitter': False, 'scaling': False, 'permutation': False, 'randompermutation': False, 'magwarp': False, 'timewarp': False, 'windowslice': False, 'windowwarp': False, 'rotation': False, 'spawner': False, 'dtwwarp': False, 'shapedtwwarp': False, 'wdba': False, 'discdtw': False, 'discsdtw': False, 'extra_tag': ''})
configs.append({'task_name': 'short_term_forecast', 'is_training': 1, 'model_id': 'm4_Quarterly', 'model': 'TimesNet', 'data': 'm4', 'root_path': './dataset/m4', 'data_path': 'ETTh1.csv', 'features': 'M', 'target': 'OT', 'freq': 'h', 'checkpoints': './checkpoints/', 'seq_len': 96, 'label_len': 48, 'pred_len': 96, 'seasonal_patterns': 'Quarterly', 'inverse': False, 'mask_rate': 0.25, 'anomaly_ratio': 0.25, 'expand': 2, 'd_conv': 4, 'top_k': 5, 'num_kernels': 6, 'enc_in': 1, 'dec_in': 1, 'c_out': 1, 'd_model': 64, 'n_heads': 8, 'e_layers': 2, 'd_layers': 1, 'd_ff': 64, 'moving_avg': 25, 'factor': 3, 'distil': True, 'dropout': 0.1, 'embed': 'timeF', 'activation': 'gelu', 'output_attention': False, 'channel_independence': 1, 'decomp_method': 'moving_avg', 'use_norm': 1, 'down_sampling_layers': 0, 'down_sampling_window': 1, 'down_sampling_method': None, 'seg_len': 48, 'num_workers': 10, 'itr': 1, 'train_epochs': 10, 'batch_size': 16, 'patience': 3, 'learning_rate': 0.001, 'des': 'Exp', 'loss': 'SMAPE', 'lradj': 'type1', 'use_amp': False, 'use_gpu': True, 'gpu': 0, 'use_multi_gpu': False, 'devices': '0,1,2,3', 'p_hidden_dims': [128, 128], 'p_hidden_layers': 2, 'use_dtw': False, 'augmentation_ratio': 0, 'seed': 2, 'jitter': False, 'scaling': False, 'permutation': False, 'randompermutation': False, 'magwarp': False, 'timewarp': False, 'windowslice': False, 'windowwarp': False, 'rotation': False, 'spawner': False, 'dtwwarp': False, 'shapedtwwarp': False, 'wdba': False, 'discdtw': False, 'discsdtw': False, 'extra_tag': ''})
configs.append({'task_name': 'short_term_forecast', 'is_training': 1, 'model_id': 'm4_Yearly', 'model': 'TimesNet', 'data': 'm4', 'root_path': './dataset/m4', 'data_path': 'ETTh1.csv', 'features': 'M', 'target': 'OT', 'freq': 'h', 'checkpoints': './checkpoints/', 'seq_len': 96, 'label_len': 48, 'pred_len': 96, 'seasonal_patterns': 'Yearly', 'inverse': False, 'mask_rate': 0.25, 'anomaly_ratio': 0.25, 'expand': 2, 'd_conv': 4, 'top_k': 5, 'num_kernels': 6, 'enc_in': 1, 'dec_in': 1, 'c_out': 1, 'd_model': 16, 'n_heads': 8, 'e_layers': 2, 'd_layers': 1, 'd_ff': 32, 'moving_avg': 25, 'factor': 3, 'distil': True, 'dropout': 0.1, 'embed': 'timeF', 'activation': 'gelu', 'output_attention': False, 'channel_independence': 1, 'decomp_method': 'moving_avg', 'use_norm': 1, 'down_sampling_layers': 0, 'down_sampling_window': 1, 'down_sampling_method': None, 'seg_len': 48, 'num_workers': 10, 'itr': 1, 'train_epochs': 10, 'batch_size': 16, 'patience': 3, 'learning_rate': 0.001, 'des': 'Exp', 'loss': 'SMAPE', 'lradj': 'type1', 'use_amp': False, 'use_gpu': True, 'gpu': 0, 'use_multi_gpu': False, 'devices': '0,1,2,3', 'p_hidden_dims': [128, 128], 'p_hidden_layers': 2, 'use_dtw': False, 'augmentation_ratio': 0, 'seed': 2, 'jitter': False, 'scaling': False, 'permutation': False, 'randompermutation': False, 'magwarp': False, 'timewarp': False, 'windowslice': False, 'windowwarp': False, 'rotation': False, 'spawner': False, 'dtwwarp': False, 'shapedtwwarp': False, 'wdba': False, 'discdtw': False, 'discsdtw': False, 'extra_tag': ''})
configs.append({'task_name': 'short_term_forecast', 'is_training': 1, 'model_id': 'm4_Monthly', 'model': 'TimesNet', 'data': 'm4', 'root_path': './dataset/m4', 'data_path': 'ETTh1.csv', 'features': 'M', 'target': 'OT', 'freq': 'h', 'checkpoints': './checkpoints/', 'seq_len': 96, 'label_len': 48, 'pred_len': 96, 'seasonal_patterns': 'Monthly', 'inverse': False, 'mask_rate': 0.25, 'anomaly_ratio': 0.25, 'expand': 2, 'd_conv': 4, 'top_k': 5, 'num_kernels': 6, 'enc_in': 1, 'dec_in': 1, 'c_out': 1, 'd_model': 32, 'n_heads': 8, 'e_layers': 2, 'd_layers': 1, 'd_ff': 32, 'moving_avg': 25, 'factor': 3, 'distil': True, 'dropout': 0.1, 'embed': 'timeF', 'activation': 'gelu', 'output_attention': False, 'channel_independence': 1, 'decomp_method': 'moving_avg', 'use_norm': 1, 'down_sampling_layers': 0, 'down_sampling_window': 1, 'down_sampling_method': None, 'seg_len': 48, 'num_workers': 10, 'itr': 1, 'train_epochs': 10, 'batch_size': 16, 'patience': 3, 'learning_rate': 0.001, 'des': 'Exp', 'loss': 'SMAPE', 'lradj': 'type1', 'use_amp': False, 'use_gpu': True, 'gpu': 0, 'use_multi_gpu': False, 'devices': '0,1,2,3', 'p_hidden_dims': [128, 128], 'p_hidden_layers': 2, 'use_dtw': False, 'augmentation_ratio': 0, 'seed': 2, 'jitter': False, 'scaling': False, 'permutation': False, 'randompermutation': False, 'magwarp': False, 'timewarp': False, 'windowslice': False, 'windowwarp': False, 'rotation': False, 'spawner': False, 'dtwwarp': False, 'shapedtwwarp': False, 'wdba': False, 'discdtw': False, 'discsdtw': False, 'extra_tag': ''})


for config in configs:
    config = Namespace(**config)
    config.pred_len = M4Meta.horizons_map[config.seasonal_patterns]  # Up to M4 config
    config.seq_len = 2 * config.pred_len  # input_len = 2*pred_len
    config.label_len = config.pred_len
    config.frequency_map = M4Meta.frequency_map[config.seasonal_patterns]

    model = TimesNet(config)
    pretrained_model = TimesNet(config)

    (train_set_o, _) = (get_data(config, 'train'))
    earlystopping = EarlyStopping('val_loss', patience=3)
    earlystopping_ft = EarlyStopping('val_loss', patience=3)
    logger = TensorBoardLogger("tb_logs", name='{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_expand{}_dc{}_fc{}_eb{}_dt{}_{}'.format(
                config.task_name,
                config.model_id,
                config.model,
                config.data,
                config.features,
                config.seq_len,
                config.label_len,
                config.pred_len,
                config.d_model,
                config.n_heads,
                config.e_layers,
                config.d_layers,
                config.d_ff,
                config.expand,
                config.d_conv,
                config.factor,
                config.embed,
                config.distil,
                config.des))
    ftlogger = TensorBoardLogger("tb_logs", name='pretrained_{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_expand{}_dc{}_fc{}_eb{}_dt{}_{}'.format(
                config.task_name,
                config.model_id,
                config.model,
                config.data,
                config.features,
                config.seq_len,
                config.label_len,
                config.pred_len,
                config.d_model,
                config.n_heads,
                config.e_layers,
                config.d_layers,
                config.d_ff,
                config.expand,
                config.d_conv,
                config.factor,
                config.embed,
                config.distil,
                config.des))
    # wandb_logger = WandbLogger(log_model="all", project="PW24", name='{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_expand{}_dc{}_fc{}_eb{}_dt{}_{}'.format(
    #             config.task_name,
    #             config.model_id,
    #             config.model,
    #             config.data,
    #             config.features,
    #             config.seq_len,
    #             config.label_len,
    #             config.pred_len,
    #             config.d_model,
    #             config.n_heads,
    #             config.e_layers,
    #             config.d_layers,
    #             config.d_ff,
    #             config.expand,
    #             config.d_conv,
    #             config.factor,
    #             config.embed,
    #             config.distil,
    #             config.des))
    # pretrain_wandb_logger = WandbLogger(log_model="all", project="PW24", name='pretrained_{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_expand{}_dc{}_fc{}_eb{}_dt{}_{}'.format(
    #             config.task_name,
    #             config.model_id,
    #             config.model,
    #             config.data,
    #             config.features,
    #             config.seq_len,
    #             config.label_len,
    #             config.pred_len,
    #             config.d_model,
    #             config.n_heads,
    #             config.e_layers,
    #             config.d_layers,
    #             config.d_ff,
    #             config.expand,
    #             config.d_conv,
    #             config.factor,
    #             config.embed,
    #             config.distil,
    #             config.des))

    
    # use 20% of training data for validation
    train_set_size = int(len(train_set_o) * 0.8)
    valid_set_size = len(train_set_o) - train_set_size

    # split the train set into two
    seed = torch.Generator().manual_seed(42)
    train_set, valid_set = torch.utils.data.random_split(train_set_o, [train_set_size, valid_set_size], generator=seed)
    fttrain_set, ftvalid_set = torch.utils.data.random_split(train_set_o, [train_set_size, valid_set_size], generator=seed)
    train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            drop_last=False)
    test_loader = torch.utils.data.DataLoader(
            valid_set,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            drop_last=False)
    # fttrain_loader = copy.deepcopy(train_loader)
    # fttest_loader = copy.deepcopy(test_loader)
    fttrain_loader = torch.utils.data.DataLoader(
            fttrain_set,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            drop_last=False)
    fttest_loader = torch.utils.data.DataLoader(
            ftvalid_set,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            drop_last=False)
    pretrain_dataset = StockDataset(tickers=["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "JPM"], interval="5m", period="1mo", seq_len=config.seq_len, label_len=config.label_len, pred_len=config.pred_len)
    
    pretrain_loader = torch.utils.data.DataLoader(
            valid_set,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            drop_last=False
    )

    trainer = pyl.Trainer(callbacks=[earlystopping], max_epochs=10, logger=logger)
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=test_loader)

    ft_trainer = pyl.Trainer(callbacks=[earlystopping_ft], max_epochs=10, logger=ftlogger)
    pretrain_trainer = pyl.Trainer(max_epochs=3)
    pretrain_trainer.fit(model=pretrained_model, train_dataloaders=pretrain_loader)
    ft_trainer.fit(model=pretrained_model, train_dataloaders=fttrain_loader, val_dataloaders=fttest_loader)
    