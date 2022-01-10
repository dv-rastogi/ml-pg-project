import pandas as pd
import torch.utils.data as Data
import torch
import numpy as np

class StockMarketData(Data.Dataset):
    def __init__(self, markets, train = True, split = 0.2, target_market = "NASDAQ", days = 60):
        self.days = days
        self.target_market = target_market
        self.markets = markets
        self.market_data = {}
        self.num_rows = 0
        self.features = 0
        for market_name in ["DJI", "NASDAQ", "NYSE", "RUSSELL", "S&P"]:
            df = pd.read_csv(f"datasets/data/Processed_{market_name}.csv")
            df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
            df = df.sort_values(by='Date')
            df = df.drop(columns=['Name', 'Date'])
            df = df.fillna(0)
            self.features = df.shape[1]
            num_rows_init = df.shape[0]
            u = df.iloc[0:int(num_rows_init*(1-split))]
            self.mean = u.mean()
            self.std = u.std()
            if not train:
                u = df.iloc[int(num_rows_init*(1-split)) - self.days:]
            self.num_rows = u.shape[0]
            u = (u - self.mean)/self.std
            self.market_data[market_name] = u
    
    def get_label(self, idx):
        return 1 if float(self.market_data[self.target_market].iloc[idx + 1]["Close"]) \
                    > float(self.market_data[self.target_market].iloc[idx]["Close"]) else 0
    
    def __getitem__(self, idx):
        end_idx = self.days + idx - 1
        lab = self.get_label(end_idx)
        final_tensor = []
        for a in self.markets:
            df = self.market_data[a]
            market_tens = torch.reshape(torch.from_numpy(np.array(df.iloc[idx:end_idx+1])), (self.features, self.days, 1))
            final_tensor.append(market_tens)
        return torch.cat(final_tensor, dim = 2).float(), lab

    
    def __len__(self):
        return self.num_rows - self.days

class StockMarketDataReg(Data.Dataset):
    def __init__(self, train = True, split = 0.2, target_market = "NASDAQ", days = 60):
        self.days = days
        df = pd.read_csv(f"datasets/data/Processed_{target_market}.csv")
        df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
        df = df.sort_values(by='Date')
        df = df.drop(columns=['Name', 'Date'])
        df = df.fillna(0)
        self.features = df.shape[1]
        num_rows_init = df.shape[0]
        u = df.iloc[0:int(num_rows_init*(1-split))]
        self.mean = u.mean()
        self.std = u.std()
        if not train:
            u = df.iloc[int(num_rows_init*(1-split)) - self.days:]
        self.num_rows = u.shape[0]
        u = (u - self.mean)/self.std
        self.market_data = u
    
    def get_label(self, idx):
        return torch.tensor([float(self.market_data.iloc[idx + 1]["Close"])])
    
    def __getitem__(self, idx):
        end_idx = self.days + idx - 1
        lab = self.get_label(end_idx)
        df = self.market_data
        market_tens = torch.reshape(torch.from_numpy(np.array(df.iloc[idx:end_idx+1])), (self.features, self.days))
        return market_tens.float(), lab
    
    def __len__(self):
        return self.num_rows - self.days
    
    def var(self):
        return self.std['Close']**2

class StockMarketDataTimeSeries(Data.Dataset):
    def __init__(self, train = True, split = 0.2, target_market = "NASDAQ", days = 60):
        self.days = days
        df = pd.read_csv(f"datasets/data/Processed_{target_market}.csv")
        df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
        df = df.sort_values(by='Date')
        df = df.drop(columns=['Name', 'Date'])
        df = df.fillna(0)
        self.features = df.shape[1]
        num_rows_init = df.shape[0]
        u = df.iloc[0:int(num_rows_init*(1-split))]
        self.max = u.max()
        # self.min = u.min()
        if not train:
            u = df.iloc[int(num_rows_init*(1-split)) - self.days:]
        self.num_rows = u.shape[0]
        u = u/self.max
        self.market_data = u
    
    def get_label(self, idx):
        return torch.tensor([float(self.market_data.iloc[idx]["Close"])])
    
    def __getitem__(self, idx):
        end_idx = self.days + idx - 1
        lab = self.get_label(end_idx)
        df = self.market_data
        close_tens = torch.reshape(torch.from_numpy(np.array(df.iloc[idx:end_idx+1]["Close"])), (self.days, 1))
        time_stamp = torch.reshape(torch.from_numpy(np.linspace(0, self.days*0.2 ,num = self.days)), (self.days, 1))
        return close_tens.float(), time_stamp.float(), lab
    
    def __len__(self):
        return self.num_rows - self.days
    
    def var(self):
        return self.market_data.std()['Close']**2

if __name__ == "__main__":
    dataset = StockMarketData(train = True, split = 0.2, target_market = "NASDAQ", days = 60)
    u = dataset.__len__() - 1
    a, b = dataset.__getitem__(u)
    print(a.size())
    print(b)