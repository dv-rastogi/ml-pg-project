from os import listdir
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

prices = {}
for fn in listdir('.'):
    if fn == 'plot.py':
        continue
    market = fn[fn.find('_') + 1: fn.rfind('.')]
    df = pd.read_csv(fn)    
    df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
    df = df.sort_values(by='Date')
    prices[market] = list(df['Close'])
    print(f'* {market} details parsed!')

years = list(map(str, df['Date'].dt.to_period('Y').unique()))
spacing = np.arange(len(years)) * len(df['Date']) / len(years)

for m in prices:
    plt.plot(range(len(prices[m])), prices[m], label=m)
plt.xticks(spacing, years, rotation=90)
plt.xlabel('Year')
plt.ylabel('Closing price')
plt.title('Closing price trend')
plt.legend()
plt.show()