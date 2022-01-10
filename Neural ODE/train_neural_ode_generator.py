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
optim = torch.optim.Adam(vae.parameters(), betas=(0.9, 0.999), lr=0.001)
dataset_train = StockMarketDataTimeSeries(train = True, split = 0.2, target_market = FLAGS.target_market, days = FLAGS.num_days)
dataloader_train = Data.DataLoader(dataset_train, batch_size = 100, shuffle = True, num_workers = 2)

vae.train()

for epoch_idx in tqdm(range(FLAGS.num_epochs)):
    for x, t, _ in dataloader_train:
        optim.zero_grad()
        x = torch.transpose(x, 0, 1)
        t = torch.transpose(t, 0, 1)
        x, t = x.cuda(), t.cuda()

        max_len = np.random.choice([90, 95, 101])
        permutation = np.random.permutation(t.shape[0])
        np.random.shuffle(permutation)
        permutation = np.sort(permutation[:max_len])

        x, t = x[permutation], t[permutation]

        x_p, z, z_mean, z_log_var = vae(x, t)
        noise_std = 0.02
        kl_loss = -0.5 * torch.sum(1 + z_log_var - z_mean**2 - torch.exp(z_log_var), -1)
        loss = 0.5 * ((x-x_p)**2).sum(-1).sum(0) / noise_std**2 + kl_loss
        loss = torch.mean(loss)
        loss /= max_len
        loss.backward()
        optim.step()
    torch.save(vae.state_dict(), f"{FLAGS.target_market}_generative.pt")