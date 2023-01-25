import pandas as pd
import datetime
import yfinance as yf
import backtrader as bt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import warnings

warnings.filterwarnings("ignore")

# In this example, the best strategy in terms of performance is WR . The ranking of strategies in base of performance follows:

# WR (7.03%): Worst Scenario or Minimax Model.
# SPY (6.53%): Buy and Hold SPY.
# CVaR (5.73%): Conditional Value at Risk.
# MV (5.68%): Mean Variance.
# CDaR (4.60%): Conditional Drawdown at Risk.
# On the other hand, the best strategy in terms of Sharpe Ratio is MV . The ranking of strategies in base of Sharpe Ratio follows:

# MV (0.701): Mean Variance.
# CVaR (0.694): Conditional Value at Risk.
# WR (0.681): Worst Scenario or Minimax Model.
# SPY (0.679): Buy and Hold SPY.
# CDaR (0.622): Conditional Drawdown at Risk.

# Date range
start = '2010-01-01'
end = '2020-12-31'

# Tickers of assets
# assets = ['JCI', 'TGT', 'CMCSA', 'CPB', 'MO', 'APA', 'MMC', 'JPM',
#           'ZION', 'PSA', 'BAX', 'BMY', 'LUV', 'PCAR', 'TXT', 'TMO',
#           'DE', 'MSFT', 'HPQ', 'SEE', 'VZ', 'CNP', 'NI', 'T', 'BA','SPY']
assets = ['JCI', 'TGT', 'CMCSA', 'CPB', 'MO','SPY']
assets.sort()
assets.sort()

# Downloading data
prices = yf.download(assets, start=start, end=end)
print(prices.head())
prices = prices.dropna()

############################################################
# Showing data
############################################################

print(prices.head())

############################################################
# Defining the backtest function 
############################################################

def backtest(datas, strategy, start, end, plot=True, **kwargs):
    cerebro = bt.Cerebro()

    # Here we add transaction costs and other broker costs
    cerebro.broker.setcash(1000000.0)
    cerebro.broker.setcommission(commission=0.005) # Commission 0.5%
    cerebro.broker.set_slippage_perc(0.005, # Slippage 0.5%
                                     slip_open=True,
                                     slip_limit=True,
                                     slip_match=True,
                                     slip_out=False)
    for data in datas:
        cerebro.adddata(data)

    # Here we add the indicators that we are going to store
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, riskfreerate=0.0)
    cerebro.addanalyzer(bt.analyzers.Returns)
    cerebro.addanalyzer(bt.analyzers.DrawDown)
    cerebro.addstrategy(strategy, **kwargs)
    cerebro.addobserver(bt.observers.Value)
    cerebro.addobserver(bt.observers.DrawDown)
    results = cerebro.run(stdstats=False)
    if plot:
        # cerebro.plot(iplot=False, start=start, end=end)
        cerebro.plot(iplot=True, start=start, end=end)
    return (results[0].analyzers.drawdown.get_analysis()['max']['drawdown'],
            results[0].analyzers.returns.get_analysis()['rnorm100'],
            results[0].analyzers.sharperatio.get_analysis()['sharperatio'])

############################################################
# Create objects that contain the prices of assets
############################################################

# Creating Assets bt.feeds
assets_prices = []
for i in assets:
    if i != 'SPY':
        prices_ = prices.drop(columns='Adj Close').loc[:, (slice(None), i)].dropna()
        prices_.columns = ['Close', 'High', 'Low', 'Open', 'Volume']
        assets_prices.append(bt.feeds.PandasData(dataname=prices_, plot=True))

# Creating Benchmark bt.feeds        
prices_ = prices.drop(columns='Adj Close').loc[:, (slice(None), 'SPY')].dropna()
prices_.columns = ['Close', 'High', 'Low', 'Open', 'Volume']
benchmark = bt.feeds.PandasData(dataname=prices_, plot=True)

print(prices_.head())

############################################################
# Building the Buy and Hold strategy
############################################################

class BuyAndHold(bt.Strategy):

    def __init__(self):
        self.counter = 0

    def next(self):
        if self.counter >= 1004:
            if self.getposition(self.data).size == 0:
                self.order_target_percent(self.data, target=0.99)
        self.counter += 1 


############################################################
# Run the backtest for the selected period
############################################################
mpl.rcParams["figure.figsize"] = (30.0,24.0) # (w, h)
mpl.rcParams['font.sans-serif'] = 'arial'      # set font
mpl.rcParams['axes.unicode_minus'] = False
mpl.rcParams['lines.linestyle'] = '-'
mpl.rcParams['lines.linewidth'] = 1.5
# plt.plot() # We need to do this to avoid errors in inline plot
# plt.show()

start = 1004
end = prices.shape[0] - 1

dd, cagr, sharpe = backtest([benchmark],
                            BuyAndHold,
                            start=start,
                            end=end,
                            plot=True)

############################################################
# Show Buy and Hold Strategy Stats 
############################################################

print(f"Max Drawdown: {dd:.2f}%")
print(f"CAGR: {cagr:.2f}%")
print(f"Sharpe: {sharpe:.3f}")


############################################################
# Calculate assets returns
############################################################

pd.options.display.float_format = '{:.4%}'.format

data = prices.loc[:, ('Adj Close', slice(None))]
data.columns = assets
data = data.drop(columns=['SPY']).dropna()
returns = data.pct_change().dropna()
print(returns.head())

############################################################
# Selecting Dates for Rebalancing
############################################################

# Selecting last day of month of available data
index = returns.groupby([returns.index.year, returns.index.month]).tail(1).index
index_2 = returns.index

# Quarterly Dates
index = [x for x in index if float(x.month) % 3.0 == 0 ] 

# Dates where the strategy will be backtested
index_ = [index_2.get_loc(x) for x in index if index_2.get_loc(x) > 1000]

############################################################
#Building Constraints
############################################################

asset_classes = {'Assets': ['JCI','TGT','CMCSA','CPB','MO'], 
                 'Industry': ['Consumer Discretionary','Consumer Discretionary',
                              'Consumer Discretionary', 'Consumer Staples',
                              'Consumer Staples']}

# asset_classes = {'Assets': ['JCI','TGT','CMCSA','CPB','MO','APA','MMC','JPM',
#                             'ZION','PSA','BAX','BMY','LUV','PCAR','TXT','TMO',
#                             'DE','MSFT','HPQ','SEE','VZ','CNP','NI','T','BA'], 
#                  'Industry': ['Consumer Discretionary','Consumer Discretionary',
#                               'Consumer Discretionary', 'Consumer Staples',
#                               'Consumer Staples','Energy','Financials',
#                               'Financials','Financials','Financials',
#                               'Health Care','Health Care','Industrials','Industrials',
#                               'Industrials','Health care','Industrials',
#                               'Information Technology','Information Technology',
#                               'Materials','Telecommunications Services','Utilities',
#                               'Utilities','Telecommunications Services','Financials']}

asset_classes = pd.DataFrame(asset_classes)
asset_classes = asset_classes.sort_values(by=['Assets'])

constraints = {'Disabled': [False, False, False],
               'Type': ['All Assets', 'All Classes', 'All Classes'],
               'Set': ['', 'Industry', 'Industry'],
               'Position': ['', '', ''],
               'Sign': ['<=', '<=', '>='],
               'Weight': [0.10, 0.20, 0.03],
               'Type Relative': ['', '', ''],
               'Relative Set': ['', '', ''],
               'Relative': ['', '', ''],
               'Factor': ['', '', '']}

constraints = pd.DataFrame(constraints)

print(constraints)

############################################################
# Building constraint matrixes for Riskfolio Lib
############################################################

# A, B = rp.assets_constraints(constraints, asset_classes)


############################################################
# Building a loop that estimate optimal portfolios on
# rebalancing dates
############################################################

models = {}

# rms = ['MV', 'MAD', 'MSV', 'FLPM', 'SLPM',
#        'CVaR', 'WR', 'MDD', 'ADD', 'CDaR']

rms = ['MV', 'CVaR', 'WR', 'CDaR']

for j in rms:
    
    weights = pd.DataFrame([])

    for i in index_:
        Y = returns.iloc[i-1000:i,:] # taking last 4 years (250 trading days per year)

        # # Building the portfolio object
        # port = rp.Portfolio(returns=Y)
        
        # # Add portfolio constraints
        # port.ainequality = A
        # port.binequality = B
        
        # Calculating optimum portfolio

        # Select method and estimate input parameters:

        # method_mu='hist' # Method to estimate expected returns based on historical data.
        # method_cov='hist' # Method to estimate covariance matrix based on historical data.

        # port.assets_stats(method_mu=method_mu, method_cov=method_cov, d=0.94)
        
        # Estimate optimal portfolio:
        
        # port.solvers = ['MOSEK']
        # port.alpha = 0.05
        # model='Classic' # Could be Classic (historical), BL (Black Litterman) or FM (Factor Model)
        # rm = j # Risk measure used, this time will be variance
        # obj = 'Sharpe' # Objective function, could be MinRisk, MaxRet, Utility or Sharpe
        # hist = True # Use historical scenarios for risk measures that depend on scenarios
        # rf = 0 # Risk free rate
        # l = 0 # Risk aversion factor, only useful when obj is 'Utility'

        # w = port.optimization(model=model, rm=rm, obj=obj, rf=rf, l=l, hist=hist)

        if w is None:
            w = weights.tail(1).T
        weights = pd.concat([weights, w.T], axis = 0)
    
    models[j] = weights.copy()
    models[j].index = index_

############################################################
# Building the Asset Allocation Class
############################################################

class AssetAllocation(bt.Strategy):

    def __init__(self):

        j = 0
        for i in assets:
            setattr(self, i, self.datas[j])
            j += 1
        
        self.counter = 0
        
    def next(self):
        if self.counter in weights.index.tolist():
            for i in assets:
                w = weights.loc[self.counter, i]
                self.order_target_percent(getattr(self, i), target=w)
        self.counter += 1

############################################################
# Backtesting Mean Variance Strategy
############################################################

assets = returns.columns.tolist()
weights = models['MV']

dd, cagr, sharpe = backtest(assets_prices,
                            AssetAllocation,
                            start=start,
                            end=end,
                            plot=True)

                            ############################################################
# Show Mean Variance Strategy Stats 
############################################################

print(f"Max Drawdown: {dd:.2f}%")
print(f"CAGR: {cagr:.2f}%")
print(f"Sharpe: {sharpe:.3f}")

############################################################
# Plotting the composition of the last MV portfolio
############################################################

w = pd.DataFrame(models['MV'].iloc[-1,:])

# We need matplotlib >= 3.3.0 to use this function
#ax = rp.plot_pie(w=w, title='Sharpe Mean Variance', others=0.05, nrow=25, cmap = "tab20",
#                 height=6, width=10, ax=None)

# w.plot.pie(subplots=True, figsize=(8, 8))
# w.show()

############################################################
# Plotting the composition of the last CVaR portfolio
############################################################

w = pd.DataFrame(models['CVaR'].iloc[-1,:])

# We need matplotlib >= 3.3.0 to use this function
#ax = rp.plot_pie(w=w, title='Sharpe Mean CVaR', others=0.05, nrow=25, cmap = "tab20",
#                 height=6, width=10, ax=None)

# w.plot.pie(subplots=True, figsize=(8, 8))
# w.show()
############################################################
# Composition per Industry
############################################################

w_classes = pd.concat([asset_classes.set_index('Assets'), w], axis=1)
w_classes = w_classes.groupby(['Industry']).sum()
w_classes.columns = ['weights']
print(w_classes)

############################################################
# Backtesting Mean Worst Realization Strategy
############################################################

assets = returns.columns.tolist()
weights = models['WR']

dd, cagr, sharpe = backtest(assets_prices,
                            AssetAllocation,
                            start=start,
                            end=end,
                            plot=True)

############################################################
# Show Worst Realization Strategy Stats 
############################################################

print(f"Max Drawdown: {dd:.2f}%")
print(f"CAGR: {cagr:.2f}%")
print(f"Sharpe: {sharpe:.3f}")

# ############################################################
# # Plotting the composition of the last WR portfolio
# ############################################################

# w = pd.DataFrame(models['WR'].iloc[-1,:])

# # We need matplotlib >= 3.3.0 to use this function
# #ax = rp.plot_pie(w=w, title='Sharpe Mean WR', others=0.05, nrow=25, cmap = "tab20",
# #                 height=6, width=10, ax=None)

# # w.pie(subplots=True, figsize=(8, 8))
# # w.show()
# ############################################################
# # Composition per Industry
# ############################################################

# w_classes = pd.concat([asset_classes.set_index('Assets'), w], axis=1)
# w_classes = w_classes.groupby(['Industry']).sum()
# w_classes.columns = ['weights']
# print(w_classes)

# ############################################################
# # Backtesting Mean CDaR Strategy
# ############################################################

# assets = returns.columns.tolist()
# weights = models['CDaR']

# dd, cagr, sharpe = backtest(assets_prices,
#                             AssetAllocation,
#                             start=start,
#                             end=end,
#                             plot=True)

# ############################################################
# # Show CDaR Strategy Stats 
# ############################################################

# print(f"Max Drawdown: {dd:.2f}%")
# print(f"CAGR: {cagr:.2f}%")
# print(f"Sharpe: {sharpe:.3f}")

# ############################################################
# # Plotting the composition of the last CDaR portfolio
# ############################################################

# w = pd.DataFrame(models['CDaR'].iloc[-1,:])

# # We need matplotlib >= 3.3.0 to use this function
# #ax = rp.plot_pie(w=w, title='Sharpe Mean CDaR', others=0.05, nrow=25, cmap = "tab20",
# #                 height=6, width=10, ax=None)

# # w.plot.pie(subplots=True, figsize=(8, 8))
# # w.show()
# ############################################################
# # Composition per Industry
# ############################################################

# w_classes = pd.concat([asset_classes.set_index('Assets'), w], axis=1)
# w_classes = w_classes.groupby(['Industry']).sum()
# w_classes.columns = ['weights']
# print(w_classes)



