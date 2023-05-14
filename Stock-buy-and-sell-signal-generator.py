#!/usr/bin/env python
# coding: utf-8

# <center><h1>🤑 Stock Buy Sell signal Generator 📈💰</h1></center>

# In[1]:


# Installing Quantstats
get_ipython().system('pip install quantstats')


# In[2]:


pip install pandas-datareader==0.9.0


# In[3]:


# Importing libraries
import pandas as pd
from pandas_datareader import data
import pandas as pd
import numpy as np
import quantstats as qs
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import plotly.express as px
import yfinance as yf
from scipy import stats

import warnings
warnings.filterwarnings('ignore')


# <h3><b>Daily Returns</b></h3>

# In[5]:


# Getting daily returns for 4 different US stocks in the same time window
aapl = qs.utils.download_returns('AAPL')
aapl = aapl.loc['2010-07-01':'2023-02-10']

tsla = qs.utils.download_returns('TSLA')
tsla = tsla.loc['2010-07-01':'2023-02-10']

dis = qs.utils.download_returns('DIS')
dis = dis.loc['2010-07-01':'2023-02-10']

amd = qs.utils.download_returns('AMD')
amd = amd.loc['2010-07-01':'2023-02-10']


# In[6]:


# Converting timezone
aapl.index = aapl.index.tz_convert(None)
tsla.index = tsla.index.tz_convert(None)
dis.index = dis.index.tz_convert(None)
amd.index = amd.index.tz_convert(None)


# In[7]:


# Plotting Daily Returns for each stock
print('\nApple Daily Returns Plot:\n')
qs.plots.daily_returns(aapl)
print('\nTesla Inc. Daily Returns Plot:\n')
qs.plots.daily_returns(tsla)
print('\nThe Walt Disney Company Daily Returns Plot:\n')
qs.plots.daily_returns(dis)
print('\nAdvances Micro Devices, Inc. Daily Returns Plot:\n')
qs.plots.daily_returns(amd)


# <h3>Cumulative Returns</h3>

# In[8]:


# Plotting Cumulative Returns for each stock
print('\nApple Cumulative Returns Plot\n')
qs.plots.returns(aapl)
print('\nTesla Inc. Cumulative Returns Plot\n')
qs.plots.returns(tsla)
print('\nThe Walt Disney Company Cumulative Returns Plot\n')
qs.plots.returns(dis)
print('\nAdvances Micro Devices, Inc. Cumulative Returns Plot\n')
qs.plots.returns(amd)


# <h3>Histograms</h3>
# 

# In[9]:


# Plotting histograms
print('\nApple Daily Returns Histogram')
qs.plots.histogram(aapl, resample = 'D')

print('\nTesla Inc. Daily Returns Histogram')
qs.plots.histogram(tsla, resample = 'D')

print('\nThe Walt Disney Company Daily Returns Histogram')
qs.plots.histogram(dis, resample = 'D')

print('\nAdvances Micro Devices, Inc. Daily Returns Histogram')
qs.plots.histogram(amd, resample = 'D')


# <h3>Kurtosis</h3>

# In[10]:


# Using quantstats to measure kurtosis
print("Apple's kurtosis: ", qs.stats.kurtosis(aapl).round(2))
print("Tesla's kurtosis: ", qs.stats.kurtosis(tsla).round(2))
print("Walt Disney's kurtosis: ", qs.stats.kurtosis(dis).round(3))
print("Advances Micro Devices' kurtosis: ", qs.stats.kurtosis(amd).round(3))


# <h3>Skewness</h3>

# In[11]:


# Measuring skewness with quantstats
print("Apple's skewness: ", qs.stats.skew(aapl).round(2))
print("Tesla's skewness: ", qs.stats.skew(tsla).round(2))
print("Walt Disney's skewness: ", qs.stats.skew(dis).round(3))
print("Advances Micro Devices' skewness: ", qs.stats.skew(amd).round(3))


# <h3>Standard Deviation</h3>

# In[12]:


# Calculating Standard Deviations

print("Apple's Standard Deviation from 2010 to 2023: ", aapl.std())

print("\nTesla's Standard Deviation from 2010 to 2023: ", tsla.std())
print("\nDisney's Standard Deviation from 2010 to 2023: ", dis.std())

print("\nAMD's Standard Deviation from 2010 to 2023: ", amd.std())


# <h3>Pairplots and Correlation Matrix</h3>

# In[13]:


# Merging daily returns into one dataframe
merged_df = pd.concat([aapl, tsla, dis, amd], join = 'outer', axis = 1)
merged_df.columns = ['aapl', 'tsla', 'dis', 'amd']
merged_df # Displaying dataframe


# In[14]:


# Pairplots
sns.pairplot(merged_df, kind = 'reg')
plt.show()


# In[15]:


get_ipython().system('pip install plotly')


# In[16]:


# import required libraries
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# assuming that `merged_df` is a pandas DataFrame containing the merged data

# Correlation Matrix
corr = merged_df.corr()
mask = np.zeros_like(corr, dtype=bool)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(corr, annot=True, mask=mask)
plt.show()


# <h3>Load Data from SP500</h3>

# In[18]:


# Loading data from the SP500, the american benchmark
sp500 = qs.utils.download_returns('^GSPC')
sp500 = sp500.loc['2010-07-01':'2023-02-10']
sp500.index = sp500.index.tz_convert(None)
sp500


# In[19]:


# Removing indexes
sp500_no_index = sp500.reset_index(drop = True)
aapl_no_index = aapl.reset_index(drop = True)
tsla_no_index = tsla.reset_index(drop = True)
dis_no_index = dis.reset_index(drop = True)
amd_no_index = amd.reset_index(drop = True)


# In[20]:


sp500_no_index


# In[21]:


aapl_no_index


# In[22]:


# Fitting linear relation among Apple's returns and Benchmark
X = sp500_no_index.values.reshape(-1,1)
y = aapl_no_index.values.reshape(-1,1)

linreg = LinearRegression().fit(X, y)

beta = linreg.coef_[0]
alpha = linreg.intercept_

print('AAPL beta: ', beta.round(3))
print('\nAAPL alpha: ', alpha.round(3))


# In[23]:


# Fitting linear relation among Tesla's returns and Benchmark
X = sp500_no_index.values.reshape(-1,1)
y = tsla_no_index.values.reshape(-1,1)

linreg = LinearRegression().fit(X, y)

beta = linreg.coef_[0]
alpha = linreg.intercept_

print('TSLA beta: ', beta.round(3))
print('\nTSLA alpha: ', alpha.round(3))


# In[24]:


# Fitting linear relation among Walt Disney's returns and Benchmark
X = sp500_no_index.values.reshape(-1,1)
y = dis_no_index.values.reshape(-1,1)

linreg = LinearRegression().fit(X, y)

beta = linreg.coef_[0]
alpha = linreg.intercept_

print('Walt Disney Company beta: ', beta.round(3))
print('\nWalt Disney Company alpha: ', alpha.round(4))


# In[25]:


# Fitting linear relation among AMD's returns and Benchmark
X = sp500_no_index.values.reshape(-1,1)
y = amd_no_index.values.reshape(-1,1)

linreg = LinearRegression().fit(X, y)

beta = linreg.coef_[0]
alpha = linreg.intercept_

print('AMD beta: ', beta.round(3))
print('\nAMD alpha: ', alpha.round(4))


# <h3>Sharpe Ratio</h3>

# In[26]:


# Calculating Sharpe ratio
print("Sharpe Ratio for AAPL: ", qs.stats.sharpe(aapl).round(2))
print("Sharpe Ratio for TSLA: ", qs.stats.sharpe(tsla).round(2))
print("Sharpe Ratio for DIS: ", qs.stats.sharpe(dis).round(2))
print("Sharpe Ratio for AMD: ", qs.stats.sharpe(amd).round(2))


# <h3> Portfolio</h3>

# In[27]:


weights = [0.25, 0.25, 0.25, 0.25] # Defining weights for each stock
portfolio = aapl*weights[0] + tsla*weights[1] + dis*weights[2] + amd*weights[3] # Creating portfolio multiplying each stock for its respective weight 
portfolio # Displaying portfolio's daily returns


# In[28]:


# Generating report on portfolio performance from July 1st, 2010 to February 10th, 2023
qs.reports.full(portfolio, benchmark = sp500)


# <h3>PyPortfolioOpt</h3>
# 

# In[29]:


# installing PyPortfolioOpt
get_ipython().system('pip install pyportfolioopt')


# In[30]:


# Getting dataframes info for Stocks using yfinance
aapl_df = yf.download('AAPL', start = '2010-07-01', end = '2023-02-11')
tsla_df = yf.download('TSLA', start = '2010-07-01', end = '2023-02-11')
dis_df = yf.download('DIS', start = '2010-07-01', end = '2023-02-11')
amd_df = yf.download('AMD', start = '2010-07-01', end = '2023-02-11')


# In[32]:


aapl_df.columns = ['aapl_Open', 'aapl_High', 'aapl_Low', 'aapl_Close', 'aapl_Adj Close', 'aapl_Volume']
tsla_df.columns = ['tsla_Open', 'tsla_High', 'tsla_Low', 'tsla_Close', 'tsla_Adj Close', 'tsla_Volume']
dis_df.columns = ['dis_Open', 'dis_High', 'dis_Low', 'dis_Close', 'dis_Adj Close', 'dis_Volume']
amd_df.columns = ['amd_Open', 'amd_High', 'amd_Low', 'amd_Close', 'amd_Adj Close', 'amd_Volume']

df = pd.concat([aapl_df['aapl_Adj Close'], tsla_df['tsla_Adj Close'], dis_df['dis_Adj Close'], amd_df['amd_Adj Close']], join='outer', axis=1)
df.columns = ['aapl', 'tsla', 'dis', 'amd']
df


# In[33]:


# Importing libraries for portfolio optimization
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns


# <h3>Markowitz Mean-Variance Optimization Model</h3>

# In[34]:


# Calculating expected annualized returns and annualized sample covariance matrix
mu = expected_returns.mean_historical_return(df) #expected returns
S = risk_models.sample_cov(df) #Covariance matrix


# In[35]:


mu # Visualizng the annualized expected returns


# In[36]:


S # Visualizing the covariance matrix


# In[37]:


# Optimizing for maximal Sharpe ratio
ef = EfficientFrontier(mu, S)
weights = ef.max_sharpe() # Calculating optimal weights for Sharpe ratio maximization 

clean_weights = ef.clean_weights() # clean_weights rounds the weights and clips near-zeros

# Printing optimized weights and expected performance for portfolio
print(clean_weights)
ef.portfolio_performance(verbose = True)


# In[38]:


# Creating new portfolio with optimized weights
new_weights = [0.70828, 0.29172]
optimized_portfolio = aapl*new_weights[0] + tsla*new_weights[1]
optimized_portfolio


# In[39]:


# Displaying new reports comparing the optimized portfolio to the first portfolio constructed
qs.reports.full(optimized_portfolio, benchmark = portfolio)


# # Confidence

# In[40]:


import pandas as pd
from pandas_datareader import data


# In[41]:



#Mapping assests
assets =['AAPL','TSLA','DIS','AMD']


# In[42]:


import yfinance as yf

# Define assets and weights
assets = ["AAPL", "AMZN", "MSFT"]
weights = [0.4, 0.4, 0.2]

# Get market caps for assets
market_caps = []
for asset in assets:
    ticker = yf.Ticker(asset)
    market_cap = ticker.info["marketCap"]
    market_caps.append(market_cap)

# Calculate portfolio market cap
portfolio_market_cap = sum([w * mc for w, mc in zip(weights, market_caps)])


# In[43]:


import yfinance as yf

assets = ['AAPL', 'AMZN', 'GOOG'] # list of tickers
data = yf.download(assets, period="max") # download stock data

market_caps = []
for asset in assets:
    shares_outstanding = yf.Ticker(asset).info['sharesOutstanding']
    latest_price = data['Adj Close'][asset].iloc[-1]
    market_cap = shares_outstanding * latest_price
    market_caps.append(market_cap)

print(market_caps)


# In[44]:


market_cap


# # Technical Indicators
# 
# 

# In[45]:


# Downloading Apple Stocks
aapl = yf.download('AAPL', start = '2010-07-01', end = '2023-02-11')


# In[46]:


aapl #Displaying Data


# In[47]:


aapl.dtypes #Printing data types


# In[48]:


from plotly.subplots import make_subplots
import plotly.graph_objs as go


# In[49]:


# Plotting candlestick chart without indicators
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights = [0.7, 0.3])
fig.add_trace(go.Candlestick(x=aapl.index,
                             open=aapl['Open'],
                             high=aapl['High'],
                             low=aapl['Low'],
                             close=aapl['Adj Close'],
                             name='AAPL'),
              row=1, col=1)
# Plotting volume chart on the second row 
fig.add_trace(go.Bar(x=aapl.index,
                     y=aapl['Volume'],
                     name='Volume',
                     marker=dict(color='orange', opacity=1.0)),
              row=2, col=1)
# Plotting annotation
fig.add_annotation(text='Apple (AAPL)',
                    font=dict(color='white', size=40),
                    xref='paper', yref='paper',
                    x=0.5, y=0.65,
                    showarrow=False,
                    opacity=0.2)
# Configuring layout
fig.update_layout(title='AAPL Candlestick Chart From July 1st, 2010 to February 10th, 2023',
                  yaxis=dict(title='Price (USD)'),
                  height=1000,
                 template = 'plotly_dark')
# Configuring axes and subplots
fig.update_xaxes(rangeslider_visible=False, row=1, col=1)
fig.update_xaxes(rangeslider_visible=False, row=2, col=1)
fig.update_yaxes(title_text='Price (USD)', row=1, col=1)
fig.update_yaxes(title_text='Volume', row=2, col=1)

fig.show()


# In[50]:


# Adding Moving Averages
aapl['EMA9'] = aapl['Adj Close'].ewm(span = 9, adjust = False).mean() # Exponential 9-Period Moving Average
aapl['SMA20'] = aapl['Adj Close'].rolling(window=20).mean() # Simple 20-Period Moving Average
aapl['SMA50'] = aapl['Adj Close'].rolling(window=50).mean() # Simple 50-Period Moving Average
aapl['SMA100'] = aapl['Adj Close'].rolling(window=100).mean() # Simple 100-Period Moving Average
aapl['SMA200'] = aapl['Adj Close'].rolling(window=200).mean() # Simple 200-Period Moving Average


# Adding RSI for 14-periods 
delta = aapl['Adj Close'].diff() # Calculating delta
gain = delta.where(delta > 0,0)  # Obtaining gain values
loss = -delta.where(delta < 0,0) # Obtaining loss values
avg_gain = gain.rolling(window=14).mean() # Measuring the 14-period average gain value
avg_loss = loss.rolling(window=14).mean() # Measuring the 14-period average loss value
rs = avg_gain/avg_loss # Calculating the RS
aapl['RSI'] = 100 - (100 / (1 + rs)) # Creating an RSI column to the Data Frame 

# Adding Bollinger Bands 20-periods
aapl['BB_UPPER'] = aapl['SMA20'] + 2*aapl['Adj Close'].rolling(window=20).std() # Upper Band
aapl['BB_LOWER'] = aapl['SMA20'] - 2*aapl['Adj Close'].rolling(window=20).std() # Lower Band

# Adding ATR 14-periods
aapl['TR'] = pd.DataFrame(np.maximum(np.maximum(aapl['High'] - aapl['Low'], abs(aapl['High'] - aapl['Adj Close'].shift())), abs(aapl['Low'] - aapl['Adj Close'].shift())), index = aapl.index)
aapl['ATR'] = aapl['TR'].rolling(window = 14).mean() # Creating an ART column to the Data Frame 


# In[51]:


aapl.tail(50) # Plotting last 50 trading days with indicators


# In[52]:


# Plotting Candlestick charts with indicators
fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.05,row_heights=[0.6, 0.10, 0.10, 0.20])

# Candlestick 
fig.add_trace(go.Candlestick(x=aapl.index,
                             open=aapl['Open'],
                             high=aapl['High'],
                             low=aapl['Low'],
                             close=aapl['Adj Close'],
                             name='AAPL'),
              row=1, col=1)

# Moving Averages
fig.add_trace(go.Scatter(x=aapl.index,
                         y=aapl['EMA9'],
                         mode='lines',
                         line=dict(color='#90EE90'),
                         name='EMA9'),
              row=1, col=1)
fig.add_trace(go.Scatter(x=aapl.index,
                         y=aapl['SMA20'],
                         mode='lines',
                         line=dict(color='yellow'),
                         name='SMA20'),
              row=1, col=1)
fig.add_trace(go.Scatter(x=aapl.index,
                         y=aapl['SMA100'],
                         mode='lines',
                         line=dict(color='purple'),
                         name='SMA100'),
              row=1, col=1)
fig.add_trace(go.Scatter(x=aapl.index,
                         y=aapl['SMA200'],
                         mode='lines',
                         line=dict(color='red'),
                         name='SMA200'),
              row=1, col=1)
# Bollinger Bands
fig.add_trace(go.Scatter(x=aapl.index,
                         y=aapl['BB_UPPER'],
                         mode='lines',
                         line=dict(color='#00BFFF'),
                         name='Upper Band'),
              row=1, col=1)

fig.add_trace(go.Scatter(x=aapl.index,
                         y=aapl['BB_LOWER'],
                         mode='lines',
                         line=dict(color='#00BFFF'),
                         name='Lower Band'),
              row=1, col=1)

fig.add_annotation(text='Apple (AAPL)',
                    font=dict(color='white', size=40),
                    xref='paper', yref='paper',
                    x=0.5, y=0.65,
                    showarrow=False,
                    opacity=0.2)

# Relative Strengh Index (RSI)
fig.add_trace(go.Scatter(x=aapl.index,
                         y=aapl['RSI'],
                         mode='lines',
                         line=dict(color='#CBC3E3'),
                         name='RSI'),
              row=2, col=1)

# Adding marking lines at 70 and 30 levels
fig.add_shape(type="line",
              x0=aapl.index[0], y0=70, x1=aapl.index[-1], y1=70,
              line=dict(color="red", width=2, dash="dot"),
              row=2, col=1)

fig.add_shape(type="line",
              x0=aapl.index[0], y0=30, x1=aapl.index[-1], y1=30,
              line=dict(color="#90EE90", width=2, dash="dot"),
              row=2, col=1)

# Average True Range (ATR)
fig.add_trace(go.Scatter(x=aapl.index,
                         y=aapl['ATR'],
                         mode='lines',
                         line=dict(color='#00BFFF'),
                         name='ATR'),
              row=3, col=1)
# Volume
fig.add_trace(go.Bar(x=aapl.index,
                     y=aapl['Volume'],
                     name='Volume',
                     marker=dict(color='orange', opacity=1.0)),
              row=4, col=1)

# Layout
fig.update_layout(title='AAPL Candlestick Chart From July 1st, 2010 to February 10th, 2023',
                  yaxis=dict(title='Price (USD)'),
                  height=1000,
                 template = 'plotly_dark')

# Axes and subplots
fig.update_xaxes(rangeslider_visible=False, row=1, col=1)
fig.update_xaxes(rangeslider_visible=False, row=4, col=1)
fig.update_yaxes(title_text='Price (USD)', row=1, col=1)
fig.update_yaxes(title_text='RSI', row=2, col=1)
fig.update_yaxes(title_text='ATR', row=3, col=1)
fig.update_yaxes(title_text='Volume', row=4, col=1)

fig.show()


# In[53]:


fig.update_xaxes(rangeslider_visible=False, row=1, col=1)


# # Dividend Yield
# 
# 

# In[54]:


# AAPL indicators (obtained from https://finance.yahoo.com/quote/AAPL/key-statistics?p=AAPL)

aapl_eps = 5.89
aapl_pe_ratio = 26.12
aapl_roe = 147.94
aapl_dy = 0.60

# AMD indicators (obtained from https://finance.yahoo.com/quote/AMD/key-statistics?p=AMD)
amd_eps = 0.82
amd_pe_ratio = 96.62
amd_roe = 4.24
amd_dy = 0.00


# In[55]:


#printing data
print("\n")
print('Apple (AAPL) Fundamental Indicators:')
print('\n')
print('Earnings per Share(EPS):',aapl_eps)
print('Price-to_Earnings Ratio(P/E):', aapl_pe_ratio)
print('Return on Equity (ROE):',aapl_roe,"%")
print('Dividend Yield:',aapl_dy,"%")
print("\n")
print('AMD Fundamnetal Indicators:')
print("\n")
print("Earnings per Share (EPS):",amd_eps)
print("Price-to-Earnings Ratio (P\E):",amd_pe_ratio)
print('Return on Equity (ROE):', amd_roe,"%")
print('Dividend yield:',amd_dy,"%")
print("\n")


# # Hourly Data

# In[56]:


import datetime as dt
#Loading houry data for the EUR/USD pair
end_date = dt.datetime.now() #Defining the datetime for March 21st
start_date = end_date - dt.timedelta(days=729)#Loading ho urly data for the last 729 days
hourly_eur_usd = yf.download('EURUSD=X', start=start_date,end=end_date, interval='1h')
hourly_eur_usd


# In[57]:


get_ipython().system('pip install ta')


# In[58]:


#Calculating the RSI with the TA library
import ta
hourly_eur_usd['rsi'] = ta.momentum.RSIIndicator(hourly_eur_usd['Adj Close'], window = 14).rsi()
hourly_eur_usd


# In[59]:


# Plotting candlestick chart for hourly EUR/USD
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05,row_heights=[0.7, 0.3])
fig.add_trace(go.Candlestick(x=hourly_eur_usd.index,
                             open=hourly_eur_usd['Open'],
                             high=hourly_eur_usd['High'],
                             low=hourly_eur_usd['Low'],
                             close=hourly_eur_usd['Adj Close'],
                             name='EUR/USD'), row=1, col=1)
# Adding annotation
fig.add_annotation(text='EUR/USD 1HR',
                    font=dict(color='white', size=40),
                    xref='paper', yref='paper',
                    x=0.5, y=0.65,
                    showarrow=False,
                    opacity=0.2)
# Relative Strengh Index (RSI) Indicator
fig.add_trace(go.Scatter(x=hourly_eur_usd.index,
                         y=hourly_eur_usd['rsi'],
                         mode='lines',
                         line=dict(color='#CBC3E3'),
                         name='RSI'),
              row=2, col=1)
# Adding marking lines at 70 and 30 levels
fig.add_shape(type="line",
              x0=hourly_eur_usd.index[0], y0=70, x1=hourly_eur_usd.index[-1], y1=70,
              line=dict(color="red", width=2, dash="dot"),
              row=2, col=1)
fig.add_shape(type="line",
              x0=hourly_eur_usd.index[0], y0=30, x1=hourly_eur_usd.index[-1], y1=30,
              line=dict(color="#90EE90", width=2, dash="dot"),
              row=2, col=1)

# Layout
fig.update_layout(title='EUR/USD Hourly Candlestick Chart from 2021 to 2023',
                  yaxis=dict(title='Price'),
                  height=1000,
                 template = 'plotly_dark')

# Configuring subplots
fig.update_xaxes(rangeslider_visible=False)
fig.update_yaxes(title_text='Price', row = 1)
fig.update_yaxes(title_text='RSI', row = 2)

fig.show()


# In[60]:


#Defining the parameters for the RSI strategy
rsi_period =14
overbought =70
oversold =30

#creating a new column to hold the signals 
hourly_eur_usd['signal'] =0 # When we're not positioned ,'signal' =0

#Generating the entries
for i in range(rsi_period, len(hourly_eur_usd)):
    if hourly_eur_usd['rsi'][i] > overbought and hourly_eur_usd['rsi'][i - 1] <= overbought:
        hourly_eur_usd['signal'][i] = -1 # We sell when 'signal' = -1
    elif hourly_eur_usd['rsi'][i] < oversold and hourly_eur_usd['rsi'][i - 1] >= oversold:
        hourly_eur_usd['signal'][i] = 1 # We buy when 'signal' = 1
# Calculating the pair's daily returns
hourly_eur_usd['returns'] = hourly_eur_usd['Adj Close'].pct_change()
hourly_eur_usd['cumulative_returns'] = (1 + hourly_eur_usd['returns']).cumprod() - 1 # Total returns of the period

# Applying the signals to the returns
hourly_eur_usd['strategy_returns'] = hourly_eur_usd['signal'].shift(1) * hourly_eur_usd['returns']

# Calculating the cumulative returns of the strategy
hourly_eur_usd['cumulative_strategy_returns'] = (1 + hourly_eur_usd['strategy_returns']).cumprod() - 1

# Setting the initial capital to $100
initial_capital = 100

# Calculating total portfolio value
hourly_eur_usd['portfolio_value'] = (1 + hourly_eur_usd['strategy_returns']).cumprod() * initial_capital

# Printing the number of trades, initial capital, and final capital
num_trades = hourly_eur_usd['signal'].abs().sum()
final_capital = hourly_eur_usd['portfolio_value'].iloc[-1]
total_return = (final_capital - initial_capital) / initial_capital * 100

print('\n')
print(f"Number of trades: {num_trades}")
print(f"Initial capital: ${initial_capital}")
print(f"Final capital: ${final_capital:.2f}")
print(f"Total return: {total_return:.2f}%")
print('\n')


# Plotting the portfolio total value 
fig = go.Figure()

fig.add_trace(go.Scatter(x=hourly_eur_usd.index,
                         y=hourly_eur_usd['portfolio_value'].round(2),
                         mode='lines',
                         line=dict(color='#00BFFF'),
                         name='Portfolio Value'))


fig.update_layout(title='Portfolio - RSI Strategy on Hourly Data',
                  xaxis_title='Date',
                  yaxis_title='Value ($)',
                  template='plotly_dark',
                 height = 600)

fig.show()


# # Daily Data

# In[61]:


#Loading daily EUR/USD pair data for the last eight years
daily_eur_usd = yf.download('EURUSD=X', start ='2015-03-13',end='2023-03-13', interval='1d')


# In[62]:


daily_eur_usd


# In[63]:


# Calculating the RSI with the TA library
daily_eur_usd["rsi"] = ta.momentum.RSIIndicator(daily_eur_usd["Adj Close"], window=14).rsi()
daily_eur_usd # Displaying dataframe


# In[64]:


# Plotting candlestick chart for daily EUR/USD
fig = make_subplots(rows=2,cols=1, shared_xaxes=True,vertical_spacing=0.05,row_heights=[0.7,0.3])
fig.add_trace(go.Candlestick(x=daily_eur_usd.index,
                            open=daily_eur_usd['Open'],
                            high=daily_eur_usd['High'],
                            low=daily_eur_usd['Low'],
                            close = daily_eur_usd['Adj Close'],
                            name='EUR/USD'),
              row=1,col=1)

#Annotation 
fig.add_annotation(text='EUR/USD 1D',
                    font=dict(color='white', size=40),
                    xref='paper', yref='paper',
                    x=0.5, y=0.65,
                    showarrow=False,
                    opacity=0.2)


#Relative Strengh Index(RSI)
fig.add_trace(go.Scatter(x=daily_eur_usd.index,
                         y=daily_eur_usd['rsi'],
                         mode='lines',
                         line=dict(color='#CBC3E3'),
                         name='RSI'),
              row=2, col=1)
# Adding marking lines at 70 and 30 levels
fig.add_shape(type="line",
              x0=daily_eur_usd.index[0], y0=70, x1=daily_eur_usd.index[-1], y1=70,
              line=dict(color="red", width=2, dash="dot"),
              row=2, col=1)

fig.add_shape(type="line",
              x0=daily_eur_usd.index[0], y0=30, x1=daily_eur_usd.index[-1], y1=30,
              line=dict(color="#90EE90", width=2, dash="dot"),
              row=2, col=1)

fig.update_layout(title='EUR/USD Daily Candlestick Chart from 2015 to 2023',
                  yaxis=dict(title='Price'),
                  height=1000,
                 template = 'plotly_dark')

# Configuring subplots
fig.update_xaxes(rangeslider_visible=False)
fig.update_yaxes(title_text='Price', row = 1)
fig.update_yaxes(title_text='RSI', row = 2)

fig.show()


# In[65]:


#Defining the parameters for the RSi strategy
rsi_period =14
overbought =70
oversold =30

#creating a new column to hold the signals
daily_eur_usd['signal'] =0
#Generating entry signals
# Generating entry signals
for i in range(rsi_period, len(daily_eur_usd)):
    if daily_eur_usd['rsi'][i] > overbought and daily_eur_usd['rsi'][i - 1] <= overbought:
        daily_eur_usd['signal'][i] = -1 # sell signal
    elif daily_eur_usd['rsi'][i] < oversold and daily_eur_usd['rsi'][i - 1] >= oversold:
        daily_eur_usd['signal'][i] = 1 # buy signal
        
# Calculating total returns for the EUR/USD
daily_eur_usd['returns'] = daily_eur_usd['Adj Close'].pct_change()
daily_eur_usd['cumulative_returns'] = (1 + daily_eur_usd['returns']).cumprod() - 1

# Applying the signals to the returns
daily_eur_usd['strategy_returns'] = daily_eur_usd['signal'].shift(1) * daily_eur_usd['returns']

# Calculating the cumulative returns for the strategy
daily_eur_usd['cumulative_strategy_returns'] = (1 + daily_eur_usd['strategy_returns']).cumprod() - 1

# Setting the initial capital
initial_capital = 100

# Calculating portfolio value
daily_eur_usd['portfolio_value'] = (1 + daily_eur_usd['strategy_returns']).cumprod() * initial_capital

# Printing the number of trades, initial capital, and final capital
num_trades = daily_eur_usd['signal'].abs().sum()
final_capital = daily_eur_usd['portfolio_value'].iloc[-1]
total_return = (final_capital - initial_capital) / initial_capital * 100


print('\n')
print(f"Number of trades: {num_trades}")
print(f"Initial capital: ${initial_capital}")
print(f"Final capital: ${final_capital:.2f}")
print(f"Total return: {total_return:.2f}%")
print('\n')

# Plotting portfolio evolution
fig = go.Figure()

fig.add_trace(go.Scatter(x=daily_eur_usd.index,
                         y=daily_eur_usd['portfolio_value'].round(2),
                         mode='lines',
                         line=dict(color='#00BFFF'),
                         name='Portfolio Value'))

fig.update_layout(title='Portfolio - RSI Strategy on Daily Data',
                  xaxis_title='Date',
                  yaxis_title='Value ($)',
                  template='plotly_dark',
                 height = 600)

fig.show()


# # Weekly Data

# In[66]:


#Weekly time frame
weekly_eur_usd = yf.download('EURUSD=X',start='2015-03-13',end='2023-03-13',interval ='1wk')


# In[67]:


#Calculating the RSI with the TA library
weekly_eur_usd['rsi'] = ta.momentum.RSIIndicator(weekly_eur_usd['Adj Close'],window=14).rsi()
weekly_eur_usd


# In[68]:


#Plotting candlestick chart for weekly EUR/USD
fig =make_subplots(rows=2,cols=1,shared_xaxes=True,vertical_spacing=0.05, row_heights=[0.7,0.3])
fig.add_trace(go.Candlestick(x=weekly_eur_usd.index,
                            open =weekly_eur_usd['Open'],
                            high =weekly_eur_usd['High'],
                            low =weekly_eur_usd['Low'],
                            close=weekly_eur_usd['Adj Close'],
                            name ='EUR,USD'),row=1, col =1)

#Annotations 
fig.add_annotation(text='EUR/USD 1W',
                  font =dict(color='white',size =40),
                  xref='paper',yref='paper',
                  x=0.5,y =0.65,
                  showarrow =False,
                  opacity =0.2
                  )

#Relative Strength Index(RSI)
fig.add_trace(go.Scatter(x=weekly_eur_usd.index,
                        y=weekly_eur_usd['rsi'],
                        mode ='lines',
                        line =dict(color='#CBC3E3'),
                        name ='RSI'),
             row=2,col=1)
#Adding marking line at 70 and 30 levels
fig.add_shape(type='line',
             x0 =weekly_eur_usd.index[0],y0=70,x1=weekly_eur_usd.index[-1],y1=70,
             line =dict(color='red',width=2,dash='dot'),row=2,col=1)
fig.add_shape(type='line',
             x0=weekly_eur_usd.index[0],y0=30,x1=weekly_eur_usd.index[-1],y1=30,
             line=dict(color='#90EE90',width =2,dash='dot'),
             row=2,col=1)
fig.update_layout(title='EUR/USD Weekly Candlestick Chart from 2015 to 2023',
                 yaxis=dict(title='Price'),
                 height=1000,
                 template='plotly_dark')

#Configuring subplots
fig.update_xaxes(rangeslider_visible =False)
fig.update_yaxes(title_text='Price',row=1)
fig.update_yaxes(title_text ='RSI',row=2)
fig.show()


# In[69]:


#Defining the parameters for the RSI strategy
rsi_period =14
overbought =70
oversold =30

#Creating a new column to hold the signals
weekly_eur_usd['signal'] =0

#Generating entry signals
for i in range(rsi_period, len(weekly_eur_usd)):
    if weekly_eur_usd['rsi'][i] > overbought and weekly_eur_usd['rsi'][i - 1] <= overbought:
        weekly_eur_usd['signal'][i] = -1 # sell signal
    elif weekly_eur_usd['rsi'][i] < oversold and weekly_eur_usd['rsi'][i - 1] >= oversold:
        weekly_eur_usd['signal'][i] = 1 # buy signal
        
#Calculating total returns
weekly_eur_usd['returns'] = weekly_eur_usd['Adj Close'].pct_change()
weekly_eur_usd['cumulative_returns'] = (1+ weekly_eur_usd['returns']).cumprod() -1

#Applying the signals to the returns
weekly_eur_usd['strategy_returns'] =weekly_eur_usd['signal'].shift(1) * weekly_eur_usd['returns']

#Calculating the cumulative returns
weekly_eur_usd['cumulative_strategy_returns'] = (1+ weekly_eur_usd['strategy_returns']).cumprod() -1

#Setting the initial capital
initial_capital =100

#Calculating total portfolio value
weekly_eur_usd['portfolio_value'] =(1 + weekly_eur_usd['strategy_returns']).cumprod() * initial_capital

#Printing the number of trades, initial capital and final capital
num_trades =weekly_eur_usd['signal'].abs().sum()
final_capital =weekly_eur_usd['portfolio_value'].iloc[-1]
total_return = (final_capital - initial_capital) / initial_capital * 100


print('\n')
print(f"Number of trades:{num_trades}")
print(f"Initial capital:${initial_capital}")
print(f"Final capital:${final_capital:2f}")
print(f"Total return:{total_return:2f}%")
print('\n')

#plotting strategy returns
fig =go.Figure()
fig.add_trace(go.Scatter(x=weekly_eur_usd.index,
                        y=weekly_eur_usd['portfolio_value'].round(2),
                        mode ='lines',
                        line=dict(color ='#00BFFF'),
                        name ='Portfolio Value'))

fig.update_layout(title ='Portfolio - RSI Strategy on weekly Data',
                 xaxis_title ='Date',
                 yaxis_title ='Value($)',
                 template ='plotly_dark',
                 height =600
                 )

fig.show()


# # Hourly Data

# In[70]:


#Creating the 9-period exponential moving average with the TA library

hourly_eur_usd['ema9'] =ta.trend.ema_indicator(hourly_eur_usd['Adj Close'], window=9)

#Creating the 20 - Period exponential moving average with the TA library
hourly_eur_usd['sma20'] =ta.trend.sma_indicator(hourly_eur_usd['Adj Close'], window=20)
hourly_eur_usd[['Open', 'High','Low', 'Adj Close','ema9','sma20']]#Displaying dataframe with the moving averages


# In[71]:


#Plotting Candlestick chart with the moving averages
fig = make_subplots(rows =1, cols=1, shared_xaxes =True, vertical_spacing =0.05)
fig.add_trace(go.Candlestick(x=hourly_eur_usd.index,
                            open =hourly_eur_usd['Open'],
                            high =hourly_eur_usd['High'],
                            low=hourly_eur_usd['Low'],
                            close =hourly_eur_usd['Adj Close'],
                            name= 'EUR/USD'))

# 9 EMA
fig.add_trace(go.Scatter(x=hourly_eur_usd.index,
                        y=hourly_eur_usd['ema9'],
                        mode ='lines',
                        line =dict(color='yellow'),
                        name ='EMA 9'))

# 20 SMA
fig.add_trace(go.Scatter(x=hourly_eur_usd.index,
                        y=hourly_eur_usd['sma20'],
                        mode='lines',
                        line =dict(color ='green'),
                        name ='SMA 20'))

#Annotation
fig.add_annotation(text ='EUR/USD 1HR',
                  font =dict(color ='white', size =40),
                  xref='paper', yref='paper',
                  x=0.5, y=0.65,
                  showarrow =False,
                  opacity =0.2)

#Layout
fig.update_layout(title='EUR/USD Hourly Candlestick Chart from 2021 to 2023',
                 yaxis =dict(title='Price'),
                            height=1000,
                            template ='plotly_dark')
fig.update_xaxes(rangeslider_visible =False)
fig.show()


# In[72]:


#Defining the parameters for the moving average crossover strategy

short_ma ='ema9'
long_ma ='sma20'

#Creating a new column to hold the signals
hourly_eur_usd['signal'] = 0

# Generating the entry signals
for i in range(1, len(hourly_eur_usd)):
    if hourly_eur_usd[short_ma][i] > hourly_eur_usd[long_ma][i] and hourly_eur_usd[short_ma][i - 1] <= hourly_eur_usd[long_ma][i - 1]:
        hourly_eur_usd['signal'][i] = 1 # buy signal
    elif hourly_eur_usd[short_ma][i] < hourly_eur_usd[long_ma][i] and hourly_eur_usd[short_ma][i - 1] >= hourly_eur_usd[long_ma][i - 1]:
        hourly_eur_usd['signal'][i] = -1 # sell sign
        
# Calculating total returns
hourly_eur_usd['returns'] = hourly_eur_usd['Adj Close'].pct_change()
hourly_eur_usd['cumulative_returns'] = (1 + hourly_eur_usd['returns']).cumprod() - 1

#Appling the signals to the returns
hourly_eur_usd['strategy_returns'] = hourly_eur_usd['signal'].shift(1) * hourly_eur_usd['returns']

#Calculating the cumulative returns
hourly_eur_usd['cumulative_strategy_returns'] = (1 + hourly_eur_usd['strategy_returns']).cumprod() - 1

#Setting the initial capital
initial_capital =100

#Calculating total portfolio value
hourly_eur_usd['portfolio_value'] = (1 + hourly_eur_usd['strategy_returns']).cumprod() * initial_capital

#Printing the number of trades, initial capital and final capital
num_trades = hourly_eur_usd['signal'].abs().sum()
final_capital = hourly_eur_usd['portfolio_value'].iloc[-1]
total_return = (final_capital - initial_capital) / initial_capital * 100


print('\n')
print(f"Number of trades: {num_trades}")
print(f"Initial capital: ${initial_capital}")
print(f"Final capital: ${final_capital:.2f}")
print(f"Total return: {total_return:.2f}%")
print('\n')

#plotting strategy value
fig = go.Figure()

fig.add_trace(go.Scatter(x=hourly_eur_usd.index,
                         y=hourly_eur_usd['portfolio_value'].round(2),
                         mode='lines',
                         line=dict(color='#00BFFF'),
                         name='Portfolio Value'))

fig.update_layout(title='Portfolio - Moving Average Crossover Strategy on Hourly Data',
                  xaxis_title='Date',
                  yaxis_title='Value ($)',
                  template='plotly_dark',
                 height = 600)

fig.show()


# # Daily Data

# In[73]:


daily_eur_usd['ema9'] = ta.trend.ema_indicator(daily_eur_usd['Adj Close'], window=9)
daily_eur_usd['sma20'] = ta.trend.sma_indicator(daily_eur_usd['Adj Close'], window=20)
daily_eur_usd[['Open', 'High','Low','Adj Close', 'ema9', 'sma20']]


# In[74]:


fig = make_subplots(rows=1, cols=1, shared_xaxes =True, vertical_spacing =0.05)
fig.add_trace(go.Candlestick(x=daily_eur_usd.index,
                             open=  daily_eur_usd['Open'],
                             high = daily_eur_usd['High'],
                             low =  daily_eur_usd['Low'],
                            close = daily_eur_usd['Adj Close'],
                             name = 'EUR/USD'
                             ))
fig.add_trace(go.Scatter(x=daily_eur_usd.index,
                        y= daily_eur_usd['ema9'],
                        mode='lines',
                        line =dict(color='yellow'),
                        name ='EMA 9'
                        ))

fig.add_trace(go.Scatter(x=daily_eur_usd.index,
                        y =daily_eur_usd['sma20'],
                        mode ='lines',
                        line =dict(color='green'),
                        name ='SMA 20'
                       ))
#Annotation
fig.add_annotation(text ='EUR/USD 1D',
                  font =dict(color ='white', size =40),
                  xref ='paper', yref='paper',
                  x=0.5, y=0.65,
                  showarrow =False,
                  opacity =0.2
                  )

fig.update_layout(title ='EUR/USD Daily Candlestick Chart from 2015 to 2023',
                 yaxis =dict(title ='Price'),
                 height =1000,
                 template ='plotly_dark'
                 )

fig.update_xaxes(rangeslider_visible = False)
fig.show()



# In[75]:


#Defining the parameters for the moving average crossover strategy
short_ma = 'ema9'
long_ma  ='sma20'

#Creating a new column to hold the signals
daily_eur_usd['signal'] =0

#Generating the enrty signals
for i in range(1, len(daily_eur_usd)):
    if daily_eur_usd[short_ma][i] > daily_eur_usd[long_ma][i] and daily_eur_usd[short_ma][i-1] <= daily_eur_usd[long_ma][i-1]:
        daily_eur_usd['signal'][i] = 1 # buy signal
    elif daily_eur_usd[short_ma][i] < daily_eur_usd[long_ma][i] and daily_eur_usd[short_ma][i-1] >= daily_eur_usd[long_ma][i-1]:
        daily_eur_usd['signal'][i] =-1 # sell signal
        
# Calculating the total returns
daily_eur_usd['returns'] = daily_eur_usd['Adj Close'].pct_change()
daily_eur_usd['cumulative_returns'] = (1 + daily_eur_usd['returns']).cumprod() -1

#Applying the signals to the returns

daily_eur_usd['strategy_returns'] = daily_eur_usd['signal'].shift(1) * daily_eur_usd['returns']

#Calculate the cumulative returns
daily_eur_usd['cumulative_Strategy_returns'] = (1 + daily_eur_usd['strategy_returns']).cumprod() -1

#setting the initial capital

initial_capital =100

# Calculating the total portfolio value
daily_eur_usd['portfolio_value'] =(1+ daily_eur_usd['strategy_returns']).cumprod() * initial_capital

#Printing the number of trades , initial capital ,final capital
num_trades = daily_eur_usd['signal'].abs().sum()
final_capital = daily_eur_usd['portfolio_value'].iloc[-1]
total_return = (final_capital - initial_capital)/ initial_capital * 100

print('\n')
print(f"Number of trades:{num_trades}")
print(f"Initial capital:${initial_capital}")
print(f"Final capital:${final_capital: .2f}")
print(f"Total return:{total_return:.2f}%")
print('/n')

# Plotting strategy portfolio
fig = go.Figure()
fig.add_trace(go.Scatter(x=daily_eur_usd.index,
                        y=daily_eur_usd['portfolio_value'].round(2),
                        mode ='lines',
                        line =dict(color='#00BFFF'),
                        name ='Portfolio Value'))

fig.update_layout(title ='Portfolio -Moving Average Crossover Strategy on Daily Data',
                 xaxis_title ='Date',
                 yaxis_title ='Value ($)',
                 template ='plotly_dark',
                 )

fig.show()


# # Weekly Data

# In[76]:


weekly_eur_usd['ema9'] = ta.trend.ema_indicator(weekly_eur_usd[ 'Adj Close'], window =9)
weekly_eur_usd['sma20'] = ta.trend.sma_indicator(weekly_eur_usd['Adj Close'], window =20)
weekly_eur_usd[['Open','High','Low', 'Adj Close', 'ema9', 'sma20']]


# In[77]:


fig = make_subplots(rows=1, cols=1, shared_xaxes=True, vertical_spacing=0.05)
fig.add_trace(go.Candlestick(x=weekly_eur_usd.index,
                             open=weekly_eur_usd['Open'],
                             high=weekly_eur_usd['High'],
                             low=weekly_eur_usd['Low'],
                             close=weekly_eur_usd['Adj Close'],
                             name='EUR/USD'))

fig.add_trace(go.Scatter(x=weekly_eur_usd.index,
                         y=weekly_eur_usd['ema9'],
                         mode='lines',
                         line=dict(color='yellow'),
                         name='EMA 9'))

fig.add_trace(go.Scatter(x=weekly_eur_usd.index,
                         y=weekly_eur_usd['sma20'],
                         mode='lines',
                         line=dict(color='green'),
                         name='SMA 20'))

fig.add_annotation(text='EUR/USD 1WK',
                    font=dict(color='white', size=40),
                    xref='paper', yref='paper',
                    x=0.5, y=0.65,
                    showarrow=False,
                    opacity=0.2)

fig.update_layout(title='EUR/USD Weekly Candlestick Chart from 2015 to 2023',
                  yaxis=dict(title='Price'),
                  height=1000,
                 template = 'plotly_dark')
fig.update_xaxes(rangeslider_visible=False)
fig.show()


# In[78]:


# Defining the parameters for the moving average crossover strategy
short_ma = 'ema9'
long_ma = 'sma20'


# Creating a new column to hold the signals
weekly_eur_usd['signal'] = 0

# Generating the entry signals
for i in range(1, len(weekly_eur_usd)):
    if weekly_eur_usd[short_ma][i] > weekly_eur_usd[long_ma][i] and weekly_eur_usd[short_ma][i - 1] <= weekly_eur_usd[long_ma][i - 1]:
        weekly_eur_usd['signal'][i] = 1 # buy signal
    elif weekly_eur_usd[short_ma][i] < weekly_eur_usd[long_ma][i] and weekly_eur_usd[short_ma][i - 1] >= weekly_eur_usd[long_ma][i - 1]:
        weekly_eur_usd['signal'][i] = -1 # sell signal
        
# Calculating the total returns
weekly_eur_usd['returns'] = weekly_eur_usd['Adj Close'].pct_change()
weekly_eur_usd['cumulative_returns'] = (1 + weekly_eur_usd['returns']).cumprod() - 1


# Applying the signals to the returns
weekly_eur_usd['strategy_returns'] = weekly_eur_usd['signal'].shift(1) * weekly_eur_usd['returns']


# Calculating the cumulative returns
weekly_eur_usd['cumulative_strategy_returns'] = (1 + weekly_eur_usd['strategy_returns']).cumprod() - 1


# Setting the initial capital
initial_capital = 100

# Calculating the portfolio value
weekly_eur_usd['portfolio_value'] = (1 + weekly_eur_usd['strategy_returns']).cumprod() * initial_capital


# Printing the number of trades, initial capital, and final capital
num_trades = weekly_eur_usd['signal'].abs().sum()
final_capital = weekly_eur_usd['portfolio_value'].iloc[-1]
total_return = (final_capital - initial_capital) / initial_capital * 100

print('\n')
print(f"Number of trades: {num_trades}")
print(f"Initial capital: ${initial_capital}")
print(f"Final capital: ${final_capital:.2f}")
print(f"Total return: {total_return:.2f}%")
print('\n')


# Plotting the strategy total returns
fig = go.Figure()

fig.add_trace(go.Scatter(x=weekly_eur_usd.index,
                         y=weekly_eur_usd['portfolio_value'].round(2),
                         mode='lines',
                         line=dict(color='#00BFFF'),
                         name='Portfolio Value'))

fig.update_layout(title='Portfolio - Moving Average Crossover Strategy on Weekly Data',
                  xaxis_title='Date',
                  yaxis_title='Value ($)',
                  template='plotly_dark',
                 height = 600)

fig.show()


# In[ ]:




