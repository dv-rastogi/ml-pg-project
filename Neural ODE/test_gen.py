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

vae = ODEVAE(1, 2, 2).cuda()

vae.load_state_dict(torch.load(f"{FLAGS.target_market}_generative.pt"))

dataset_test = StockMarketDataTimeSeries(train = False, split = 0.2, target_market = FLAGS.target_market, days = FLAGS.num_days)
dataloader_test = Data.DataLoader(dataset_test, batch_size = 1, shuffle = True, num_workers = 2)
ma = dataset_test.max["Close"]

y_pred = []
y_true = []

vae.eval()

for x, t, lab in dataloader_test:
    frm, to, to_seed = 0, FLAGS.num_days, FLAGS.num_days-1
    seed_trajs = x[frm:to_seed]
    ts = t[frm:to]
    ts = ts.cuda()
    seed_trajs = seed_trajs.cuda()
    samp_trajs_p = vae.generate_with_seed(seed_trajs, ts).detach().cpu().numpy()[:, -1]
    y_pred.append(samp_trajs_p[0][0])
    y_true.append(lab[0][0])

print(ma)
print(np.mean(np.array(y_pred)))
print(np.max(np.array(y_pred)))
print(np.mean(np.array(y_true)))
print(np.max(np.array(y_true)))
print(np.var(ma*np.array(y_true)))

rmse = (ma**2*np.mean((np.array(y_pred) - np.array(y_true))**2))**0.5
mae = np.abs(ma)*np.mean(np.abs(np.array(y_pred) - np.array(y_true)))
r_squared = 1 - ((rmse)**2)/(np.var(ma*np.array(y_true)))
print(f"MSE on {FLAGS.target_market}, RMSE: {rmse}, MAE:{mae}, R-Squared:{r_squared}")