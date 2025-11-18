# ARIMA/LSTM Forecast for SGD series

Files added:
- `data_subset.csv` : provided SGD series (subset)
- `forecast_arima.py` : script to fit ARIMA and forecast
- `requirements.txt` : python dependencies

Quick run (Windows PowerShell):
```powershell
python -m pip install -r requirements.txt
python forecast_arima.py --data data_subset.csv --steps 10 --order 1,1,1 --out-csv forecast.csv --out-png forecast.png
```

Outputs:
- `forecast.csv` : forecasted values with confidence intervals
- `forecast.png` : plot of history + forecast

Notes:
- Script uses ARIMA(p,d,q) default (1,1,1). Adjust `--order` or extend script to use `pmdarima.auto_arima` for automatic selection.
- The data file uses business-day frequency and forward-fills non-trading days; change behavior as needed.

