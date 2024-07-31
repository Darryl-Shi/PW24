import yfinance as yf
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

# class StockDataset(Dataset):
#     def __init__(self, ticker, interval, period, seq_len, label_len, pred_len, scale=True):
#         self.ticker = ticker
#         self.interval = interval
#         self.period = period
#         self.seq_len = seq_len
#         self.label_len = label_len
#         self.pred_len = pred_len
#         self.scale = scale

#         self.__download_data__()
#         self.__prepare_data__()

#     def __download_data__(self):
#         self.data = yf.download(self.ticker, period=self.period, interval=self.interval)
#         self.data.reset_index(inplace=True)

#     def __prepare_data__(self):
#         self.scaler = StandardScaler()

#         # Only keep the 'Close' column for univariate data
#         self.data_y = self.data[['Close']].values

#         if self.scale:
#             self.data_y = self.scaler.fit_transform(self.data_y)
        
#         df_stamp = self.data[['Datetime']]
#         df_stamp['Datetime'] = pd.to_datetime(df_stamp.Datetime)
#         df_stamp['month'] = df_stamp.Datetime.apply(lambda row: row.month, 1)
#         df_stamp['day'] = df_stamp.Datetime.apply(lambda row: row.day, 1)
#         df_stamp['weekday'] = df_stamp.Datetime.apply(lambda row: row.weekday(), 1)
#         df_stamp['hour'] = df_stamp.Datetime.apply(lambda row: row.hour, 1)
#         self.data_stamp = df_stamp.drop(['Datetime'], 1).values

#     def __getitem__(self, index):
#         s_begin = index
#         s_end = s_begin + self.seq_len
#         r_begin = s_end - self.label_len
#         r_end = r_begin + self.label_len + self.pred_len

#         seq_x = self.data_y[s_begin:s_end]
#         seq_y = self.data_y[r_begin:r_end]
#         seq_x_mark = self.data_stamp[s_begin:s_end]
#         seq_y_mark = self.data_stamp[r_begin:r_end]

#         return seq_x, seq_y, seq_x_mark, seq_y_mark

#     def __len__(self):
#         return len(self.data_y) - self.seq_len - self.pred_len + 1

#     def inverse_transform(self, data):
#         return self.scaler.inverse_transform(data)

# Example usage:
# dataset = StockDataset(ticker='AAPL', interval='1d', period='5y', seq_len=60, label_len=20, pred_len=20)
# DataLoader can then be used to load this dataset for training/validation/testing

import yfinance as yf
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

class StockDataset(Dataset):
    def __init__(self, tickers, interval, period, seq_len, label_len, pred_len, scale=True):
        self.tickers = tickers
        self.interval = interval
        self.period = period
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.scale = scale

        self.__download_data__()
        self.__prepare_data__()

    def __download_data__(self):
        # Download data for each ticker and concatenate
        all_data = []
        for ticker in self.tickers:
            data = yf.download(ticker, period=self.period, interval=self.interval)
            data.reset_index(inplace=True)
            data['Ticker'] = ticker
            all_data.append(data)
        self.data = pd.concat(all_data)

    def __prepare_data__(self):
        self.scaler = StandardScaler()

        # Only keep the 'Close' column for univariate data
        self.data_y = self.data[['Close']].values

        if self.scale:
            self.data_y = self.scaler.fit_transform(self.data_y)
        
        df_stamp = self.data[['Datetime']]
        df_stamp['Datetime'] = pd.to_datetime(df_stamp.Datetime)
        df_stamp['month'] = df_stamp.Datetime.apply(lambda row: row.month, 1)
        df_stamp['day'] = df_stamp.Datetime.apply(lambda row: row.day, 1)
        df_stamp['weekday'] = df_stamp.Datetime.apply(lambda row: row.weekday(), 1)
        df_stamp['hour'] = df_stamp.Datetime.apply(lambda row: row.hour, 1)
        self.data_stamp = df_stamp.drop(['Datetime'], 1).values
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_y[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_y) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

# Example usage:
# tickers = ['AAPL', 'MSFT', 'GOOGL']
# dataset = StockDataset(tickers=tickers, interval='1d', period='5y', seq_len=60, label_len=20, pred_len=20)
# DataLoader can then be used to load this dataset for training/validation/testing
