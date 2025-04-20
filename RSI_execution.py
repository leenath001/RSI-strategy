import RSI_functions as RSI
import yfinance as yf

# terminal func to run while in clamshell, ensure amphetaime is on

# macair
# caffeinate -i python3 "{filepath}"

# macmini
# caffeinate -i python3 "{filepath}"

#x = RSI.RSI_breakout_backtest('RKLB',12,2025)
# (UAMY,8)

x = RSI.RSI_tradingfunc('QQQ',9)
print()
print(x[0])
print(x[1])
