import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import torch.optim as optim

from sklearn.metrics import classification_report
from tqdm import tqdm

from flags import *
# from models import *
from neural_ode_models import *
from datasets import *

def weights_init(layer):
    if isinstance(layer, nn.Conv2d):
        nn.init.kaiming_normal_(layer.weight.data, nonlinearity = "relu")
        try:
            layer.bias.data.zero_()
        except:
            pass
    elif isinstance(layer, nn.BatchNorm2d):
        layer.weight.data.normal_(1.0, 0.02)
        try:
            layer.bias.data.zero_()
        except:
            pass
    elif isinstance(layer, nn.Linear):
        layer.weight.data.normal_(0.0, 0.05)
        try:
            layer.bias.data.zero_()
        except:
            pass

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if FLAGS.all_markets:
    dataset_train = StockMarketData(markets = ["NASDAQ", "DJI", "NYSE", "S&P", "RUSSELL"], 
                                        train = True, split = 0.2, 
                                        target_market = FLAGS.target_market, 
                                        days = FLAGS.num_days)
    markets = 5
else:
    dataset_train = StockMarketData(markets = [FLAGS.target_market], train = True, split = 0.2, 
                                        target_market = FLAGS.target_market, days = FLAGS.num_days)
    markets = 1

if FLAGS.regression:
    dataset_train = StockMarketDataReg(train = True, split = 0.2, 
                                        target_market = FLAGS.target_market, 
                                        days = FLAGS.num_days)
                                        
train_dataloader = Data.DataLoader(dataset_train, batch_size = FLAGS.batch_size, 
                                        shuffle = True, num_workers = FLAGS.num_workers)

# model = CNNPred(features = FLAGS.num_features, days = FLAGS.num_days, 
#                         markets = markets)

func = ConvODEF(64)
ode = NeuralODE(func)
model = ODEStockClassifier(ode, markets)

if FLAGS.regression:
    func = ConvODEF(64)
    ode = NeuralODE(func)
    model = ODEStockRegressor(ode, markets)
    # model = CNN_LSTM(FLAGS.num_days, FLAGS.num_features, FLAGS.batch_size)

model = model.to(device)
model.apply(weights_init)

if FLAGS.regression:
    loss_fn = nn.MSELoss()
else:
    loss_fn = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=FLAGS.learning_rate, 
                        betas=(FLAGS.beta_1, FLAGS.beta_2), 
                        eps=1e-08, weight_decay=FLAGS.weight_decay, amsgrad=False)

training_loss = []

for epoch in range(1, FLAGS.num_epochs + 1):
    for inp_tensor, result in tqdm(train_dataloader, f"Training Epoch {epoch}"):
        if FLAGS.regression:
            inp_tensor.unsqueeze_(3)
        # print(inp_tensor.size())
        inp_tensor = torch.transpose(inp_tensor, 1, 3)
        inp_tensor = inp_tensor.to(device)
        result = result.to(device)
        pred = model(inp_tensor)
        loss = loss_fn(pred, result)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        training_loss.append(loss.detach().cpu().item())
    
    torch.save(model.state_dict(), FLAGS.model_save)