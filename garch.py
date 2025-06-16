import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from arch import arch_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.tsa.stattools import adfuller, acf, pacf
import pickle
from glob import glob

def prepare_data(file_path):
    df = pd.read_csv(file_path, parse_dates=['Data'])
    df.set_index('Data', inplace=True)
    df.sort_index(inplace=True)
    df = df.asfreq('B')
    df['Zamkniecie'] = df['Zamkniecie'].ffill()
    df['LogReturn'] = np.log(df['Zamkniecie'] / df['Zamkniecie'].shift(1)) * 100
    return df.dropna()

def test_stationarity(df, output_path):
    result = adfuller(df['LogReturn'].dropna())
    acf_vals = acf(df['LogReturn'].dropna(), nlags=20)
    pacf_vals = pacf(df['LogReturn'].dropna(), nlags=20)
    with open(output_path, 'w') as f:
        f.write("Test stacjonarności (ADF):\n")
        f.write(f"Statystyka ADF: {result[0]}\n")
        f.write(f"p-value: {result[1]}\n")
        f.write(f"Wartości krytyczne: {result[4]}\n\n")
        f.write("Autokorelacja (ACF):\n")
        f.write(', '.join(map(str, acf_vals)) + '\n\n')
        f.write("Częściowa autokorelacja (PACF):\n")
        f.write(', '.join(map(str, pacf_vals)) + '\n')

def split_by_periods(df):
    okresy = {
        '2013-2015_regulacje': ('2013-09-01', '2015-10-31'),
        '2015-2017_podatek_frankowicze': ('2015-11-01', '2017-12-31'),
        '2018-2019_przed_covid': ('2018-01-01', '2019-12-31'),
        '2020-2021_covid': ('2020-01-01', '2021-12-31'),
        '2022-2023_wojna_inflacja': ('2022-01-01', '2023-12-31'),
    }
    return {nazwa: df.loc[start:end].copy() for nazwa, (start, end) in okresy.items() if not df.loc[start:end].empty}

def fit_and_forecast_model(df, window, file_name, model_type, period_name, output_dir):
    df['Volatility'] = df['LogReturn'].rolling(window=window).std()
    df.dropna(inplace=True)

    y = df['Volatility'].shift(-1).dropna()
    X = df['LogReturn'].shift(1).dropna()
    common_idx = X.index.intersection(y.index)
    df_model = pd.DataFrame({'LogReturn': X.loc[common_idx], 'Volatility': y.loc[common_idx]})

    preds, dates, y_true = [], [], []
    n = len(df_model)
    start_idx = int(n * 0.8)

    for i in range(start_idx, n - 1):
        train = df_model['LogReturn'].iloc[:i]
        test_date = df_model.index[i + 1]
        true_vol = df_model['Volatility'].iloc[i + 1]
        try:
            model = arch_model(train, mean='Zero', vol=model_type, p=1, q=1, power=2.0, dist='normal')
            res = model.fit(disp='off', options={'maxiter': 5000})
            forecast = res.forecast(horizon=1)
            pred_vol = np.sqrt(forecast.variance.values[-1][0])

            preds.append(pred_vol)
            y_true.append(true_vol)
            dates.append(test_date)

            model_filename = f"model_{file_name}_{period_name}_win{window}_{model_type}.pkl"
            with open(os.path.join(output_dir, model_filename), 'wb') as f:
                pickle.dump(res, f)
        except Exception:
            continue

    pred_df = pd.DataFrame({'Date': dates, 'TrueVol': y_true, 'PredVol': preds}).set_index('Date')
    pred_df.dropna(inplace=True)

    if not pred_df.empty:
        valid_mask = (pred_df['TrueVol'] != 0) & (~pred_df['TrueVol'].isna()) & (~pred_df['PredVol'].isna())
        y_true = pred_df['TrueVol'][valid_mask]
        y_pred = pred_df['PredVol'][valid_mask]

        if not y_true.empty and not y_pred.empty:
            metrics = {
                'File': file_name,
                'Period': period_name,
                'Window': window,
                'Model': model_type,
                'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
                'MSE': mean_squared_error(y_true, y_pred),
                'MAE': mean_absolute_error(y_true, y_pred),
                'R2': r2_score(y_true, y_pred),
                'MAPE': np.mean(np.abs((y_true - y_pred) / y_true)) * 100
            }
        else:
            metrics = {
                'File': file_name,
                'Period': period_name,
                'Window': window,
                'Model': model_type,
                'RMSE': None,
                'MSE': None,
                'MAE': None,
                'R2': None,
                'MAPE': None
            }
    else:
        metrics = {
            'File': file_name,
            'Period': period_name,
            'Window': window,
            'Model': model_type,
            'RMSE': None,
            'MSE': None,
            'MAE': None,
            'R2': None,
            'MAPE': None
        }

    return [metrics]

if __name__ == '__main__':
    windows = [5, 10, 20]
    model_types = ['GARCH']
    data_dir = 'dane_wejsciowe'
    output_dir = 'WYNIKI_GARCH'
    os.makedirs(output_dir, exist_ok=True)

    results = []

    for file_path in glob(os.path.join(data_dir, '*.csv')):
        file_name = os.path.basename(file_path).replace('.csv', '')
        df_full = prepare_data(file_path)

        print(f"Przetwarzanie pliku: {file_name}")
        test_stationarity(df_full, os.path.join(output_dir, f'stats_{file_name}.txt'))

        periods = split_by_periods(df_full)

        for period_name, df_period in periods.items():
            for window in windows:
                for model_type in model_types:
                    print(f"Model: {model_type}, Okno: {window}, Okres: {period_name}")
                    metrics = fit_and_forecast_model(df_period.copy(), window, file_name, model_type, period_name, output_dir)
                    results.extend(metrics)

    results_df = pd.DataFrame(results)
    results_df.to_excel(os.path.join(output_dir, 'GARCH_METRYKI.xlsx'), index=False)

    zestawy_danych = results_df['File'].unique()
    okresy = [
        '2013-2015_regulacje',
        '2015-2017_podatek_frankowicze',
        '2018-2019_przed_covid',
        '2020-2021_covid',
        '2022-2023_wojna_inflacja'
    ]
    windows_list = [5, 10, 20]
    metryki = ['RMSE', 'MSE', 'MAE', 'R2', 'MAPE']

    final_table = pd.DataFrame()
    for file in zestawy_danych:
        for window in windows_list:
            row_block = []
            for metric in metryki:
                row = []
                for period in okresy:
                    value = results_df[(results_df['File'] == file) & (results_df['Window'] == window) & (results_df['Period'] == period) & (results_df['Model'] == 'GARCH')][metric]
                    row.append(value.values[0] if not value.empty else np.nan)
                row_block.append(row)
            block_df = pd.DataFrame(row_block, columns=[f'OKRES {i}' for i in range(1, 6)], index=metryki)

            block_df.columns = pd.MultiIndex.from_product([[file], [f'okno {window}-dniowe'], block_df.columns])
            final_table = pd.concat([final_table, block_df], axis=1)

    tabelaryczny_path = os.path.join(output_dir, 'GARCH_METRYKI_TABELARYCZNIE.xlsx')
    final_table.to_excel(tabelaryczny_path)

    print(" Wyniki zapisane w folderze:", output_dir)
