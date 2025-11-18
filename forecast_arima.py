import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import os
import argparse


def load_data(path):
    # 读取为字符串，防止文件中有重复表头或异常格式
    df = pd.read_csv(path, dtype=str, header=0, names=["DATE", "SGD"])
    # 删除像 "DATE,SGD" 这样的重复表头行
    df = df[~df['DATE'].str.upper().eq('DATE')]
    # 解析日期与数值，无法解析的会变为 NaN
    df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')
    df['SGD'] = pd.to_numeric(df['SGD'], errors='coerce')
    df = df.dropna(subset=['DATE', 'SGD'])
    # 按日期排序并去重（保留最后出现的记录）
    df = df.sort_values('DATE').drop_duplicates(subset=['DATE'], keep='last')
    df = df.set_index('DATE').sort_index()
    # 用工作日索引重建序列（会在缺失工作日处产生 NaN，后续 prepare_series 会 ffill）
    idx = pd.date_range(start=df.index.min(), end=df.index.max(), freq='B')
    s = df['SGD'].reindex(idx)
    s.index.name = 'DATE'
    return s


def prepare_series(series):
    # forward-fill any business-day gaps, but keep original index cadence
    series = series.ffill()
    return series


def fit_arima(series, order=(1,1,1)):
    model = ARIMA(series, order=order)
    fitted = model.fit()
    return fitted


def forecast_and_save(fitted, steps, out_csv, out_png):
    pred = fitted.get_forecast(steps=steps)
    mean = pred.predicted_mean
    conf = pred.conf_int()

    result = pd.DataFrame({
        'forecast': mean,
        'lower': conf.iloc[:,0],
        'upper': conf.iloc[:,1]
    })
    result.index.name = 'DATE'
    result.to_csv(out_csv)

    # plot
    plt.figure(figsize=(10,5))
    series = fitted.data.endog
    idx = pd.DatetimeIndex(fitted.data.row_labels)
    plt.plot(idx, series, label='history')
    plt.plot(result.index, result['forecast'], label='forecast')
    plt.fill_between(result.index, result['lower'], result['upper'], color='gray', alpha=0.3)
    plt.legend()
    plt.title('ARIMA Forecast')
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()

    return result


def main():
    parser = argparse.ArgumentParser(description='ARIMA forecast for SGD series')
    parser.add_argument('--data', default='data_subset.csv', help='CSV file with DATE,SGD')
    parser.add_argument('--steps', type=int, default=10, help='Number of business-day steps to forecast')
    parser.add_argument('--order', default='1,1,1', help='ARIMA order p,d,q as comma-separated')
    parser.add_argument('--out-csv', default='forecast.csv')
    parser.add_argument('--out-png', default='forecast.png')
    args = parser.parse_args()

    data_path = os.path.join(os.path.dirname(__file__), args.data)
    s = load_data(data_path)
    s = prepare_series(s)

    order = tuple(int(x) for x in args.order.split(','))
    fitted = fit_arima(s, order=order)

    print(fitted.summary())

    result = forecast_and_save(fitted, args.steps, args.out_csv, args.out_png)
    print('\nForecast saved to:', args.out_csv)
    print('Plot saved to:', args.out_png)
    print('\nForecast values:')
    print(result)


if __name__ == '__main__':
    main()
