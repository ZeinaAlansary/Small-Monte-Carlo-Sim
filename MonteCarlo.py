import datetime as dt

import matplotlib.pyplot as plt
import numpy as np
import yfinance as yf

# import data


def get_data(stocks, start, end):
    stockData = yf.download(stocks, start=start, end=end, auto_adjust=True)
    stockData = stockData["Close"]
    returns = stockData.pct_change()
    meanReturns = returns.mean()
    covMatrix = returns.cov()
    return meanReturns, covMatrix


stockList = ["CBA", "BHP", "TLS", "NAB", "WBC", "STO"]
stocks = [stock + ".AX" for stock in stockList]
endDate = dt.datetime.now()
startDate = endDate - dt.timedelta(days=300)

meanReturns, covMatrix = get_data(stocks, startDate, endDate)

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

plt.plot(portfolio_sims)
plt.ylabel("Portfolio Values ($)")
plt.xlabel("Days")
plt.title("MC simulation of a stock portfolio")
plt.show()
