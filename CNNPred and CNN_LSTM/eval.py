import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import torch.optim as optim

from sklearn.metrics import classification_report
from tqdm import tqdm

from flags import *
from models import *
from datasets import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if FLAGS.all_markets:
    dataset_test = StockMarketData(markets = ["NASDAQ", "DJI", "NYSE", "S&P", "RUSSELL"], 
                                        train = False, split = 0.2, 
                                        target_market = FLAGS.target_market, 
                                        days = FLAGS.num_days)
    markets = 5
else:
    dataset_test = StockMarketData(markets = [FLAGS.target_market], train = False, split = 0.2, target_market = FLAGS.target_market, days = FLAGS.num_days)
    markets = 1

if FLAGS.regression:
    dataset_test = StockMarketDataReg(train = False, split = 0.2, target_market = FLAGS.target_market, days = FLAGS.num_days)
    var = dataset_test.var()

test_dataloader = Data.DataLoader(dataset_test, batch_size = FLAGS.batch_size, shuffle = True, num_workers = FLAGS.num_workers)
print("F")

model = CNNPred(features = FLAGS.num_features, days = FLAGS.num_days, 
                        markets = markets)
if FLAGS.regression:
    model = CNN_LSTM(FLAGS.num_days, FLAGS.num_features, FLAGS.batch_size)
model = model.to(device)
model.load_state_dict(torch.load(FLAGS.model_save))

y_true = []
y_pred = []

with torch.no_grad():
    for inp_tensor, result in tqdm(test_dataloader, f"Testing"):
        inp_tensor = inp_tensor.to(device)
        result = result.cpu().squeeze().numpy().tolist()
        pred = model(inp_tensor)
        if not FLAGS.regression:
            pred = torch.argmax(pred, dim=1)
        pred = pred.detach().cpu().squeeze().numpy().tolist()
        
        y_pred.extend(pred)
        y_true.extend(result)

if FLAGS.regression:
    rmse = (var*np.mean((np.array(y_pred) - np.array(y_true))**2))**0.5
    mae = np.abs(var)**0.5*np.mean(np.abs(np.array(y_pred) - np.array(y_true)))
    r_squared = 1 - (rmse**2)/((var**2)*np.var(np.array(y_true)))
    print(f"MSE on {FLAGS.target_market}, RMSE: {rmse}, MAE:{mae}, R-Squared:{r_squared}")
else:
    target_names = ["0", "1"]
    metric_dict = classification_report(y_true, y_pred, 
                                            target_names = target_names, output_dict = True)
    
    average_f_score = 0
    for label in target_names:
        average_f_score += metric_dict[label]['f1-score']
    average_f_score /= len(target_names)
    
    max_f_score = 0
    for label in target_names:
        max_f_score = max(max_f_score, metric_dict[label]['f1-score'])
    
    accuracy = np.sum(np.array(y_true)==np.array(y_pred))/len(y_true)
    print(f"Average f-score for {FLAGS.target_market} input on All Markets as {FLAGS.all_markets}: {accuracy}")
    print(f"Average f-score for {FLAGS.target_market} input on All Markets as {FLAGS.all_markets}: {average_f_score}")
    print(f"Max f-score for {FLAGS.target_market} input on All Markets as {FLAGS.all_markets}: {max_f_score}")