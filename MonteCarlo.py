# importing all libraries

import datetime as dt

import matplotlib.pyplot as plt
import numpy as np
import yfinance as yf

# import data


# get_data function to get ans sort stock data from yahoo
def get_data(stocks, start, end):
    stockData = yf.download(stocks, start=start, end=end, auto_adjust=True)
    stockData = stockData["Close"]
    returns = stockData.pct_change()
    meanReturns = returns.mean()
    covMatrix = returns.cov()
    return meanReturns, covMatrix


# setting the list of stocks, start date, and end date
stockList = ["CBA", "BHP", "TLS", "NAB", "WBC", "STO"]
stocks = [stock + ".AX" for stock in stockList]
endDate = dt.datetime.now()
startDate = endDate - dt.timedelta(days=300)

# getting my mean returns from the get_data function
meanReturns, covMatrix = get_data(stocks, startDate, endDate)

# setting the weights
weights = np.random.random(len(meanReturns))
weights /= np.sum(weights)

# Monte Carlo Simulation
mc_sims = 10000
T = 1000

meanM = np.full(shape=(T, len(weights)), fill_value=meanReturns)
meanM = meanM.T

portfolio_sims = np.full(shape=(T, mc_sims), fill_value=0.0)

initialPortfolio = 10000
for m in range(mc_sims):
    Z = np.random.normal(size=(T, len(weights)))
    L = np.linalg.cholesky(covMatrix)
    dailyReturns = meanM + np.inner(L, Z)
    portfolio_sims[:, m] = (
        np.cumprod(np.inner(weights, dailyReturns.T) + 1) * initialPortfolio
    )

# creating a plot with the portfolio values and # of days since investing
plt.plot(portfolio_sims)
plt.ylabel("Portfolio Values ($)")
plt.xlabel("Days")
plt.title("MC simulation of a stock portfolio")
plt.show()
