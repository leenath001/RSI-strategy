# RSI-strategy
Functions for employing RSI trading strategy. MUST INCLUDE INDICATORS & DATAFUNCS SCRIPT TO FILE FOR TRADINGFUNC TO WORK

Goal : Strategy aims to use RSI indicator to identify levels where asset is overbought/oversold. Typically, RSI < 30 implies oversold, while >70 implies overbought. 
*  To run tradingfuncs, create an interactive brokers account and download IB Gateway (API must be simultaneously running alongside the function inside a terminal). 
*  To run autonomously, download 'Amphetamine' (mac) on app store. Also, employ caffeinate -i python3 '{filepath of execution files}' to run within terminal. Processes will run in the background while laptop/computer is inactive/closed. Trading functions run until Ctrl + C is used.

To change timeframe, see line 186 (tradingfunc) and line 35 (backtest). Allowed parameters for period and interval are included below. 

INVERVAL PARAMETERS ('period', interaval)
*  "1m" Max 7 days, only for recent data
*  "2m" Max 60 days
*  "5m" Max 60 days
*  "15m" Max 60 days
*  "30m" Max 60 days
*  "60m" Max 730 days (~2 years)
*  "90m" Max 60 days
*  "1d"

## RSI_strat.RSI_breakout(ticker,window,year)
 *  window gives period for RSI to be calculated, year calls period of data wanted for backtest
 *  Buy condition: Buy first instance that RSI < 30. Hold for all other instances following.
 *  Sell conditon: Sell first instance that RSI > 70. Do nothing for all other instances following.

## RSI_strat.RSI_tradingfunc(ticker,window)
 *  function for employing SMA strategy using interactive brokers (IB) gateway, must download/import indicators file into execution script 
 *  window gives period for RSI to be calculated (per minute basis, testing with 9periods in [9-14])
 *  function runs a while True loop. end with Ctrl + c
