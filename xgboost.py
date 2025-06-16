import pandas as pd
import numpy as np
import os
import holidays
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
from itertools import product

def prepare_data_with_sessions(file_path, window_size):
    start_date = '2010-01-01'
    end_date = '2024-12-31'
    bizdays = pd.date_range(start=start_date, end=end_date, freq='B')

    pl_holidays = holidays.Poland(years=range(2010, 2025))
    gpw_sessions = [day for day in bizdays if day not in pl_holidays]

    df = pd.read_csv(file_path, parse_dates=['Data'])
    df.set_index('Data', inplace=True)
    df.sort_index(inplace=True)
    df = df.reindex(pd.DatetimeIndex(gpw_sessions))
    df['Zamkniecie'] = df['Zamkniecie'].ffill()
    df['LogReturn'] = np.log(df['Zamkniecie'] / df['Zamkniecie'].shift(1))
    df['Volatility'] = df['LogReturn'].rolling(window=window_size).std()
    df.dropna(inplace=True)
    return df

# === Podział na okresy rynkowe ===
def split_by_periods(df):
    okresy = {
       '2013-2015_regulacje': ('2013-09-01', '2015-10-31'),
       '2015-2017_podatek_frankowicze': ('2015-11-01', '2017-12-31'),
        '2018-2019_przed_covid': ('2018-01-01', '2019-12-31'),
       '2020-2021_covid': ('2020-01-01', '2021-12-31'),
        '2022-2023_wojna_inflacja': ('2022-01-01', '2023-12-31'),
    }
    okresy_df = {}
    for nazwa, (start, end) in okresy.items():
        okres_df = df.loc[start:end].copy()
        if not okres_df.empty:
            okresy_df[nazwa] = okres_df
    return okresy_df


def create_features(X, y, seq_len):
    Xs, ys = [], []
    for i in range(seq_len, len(X)):
        Xs.append(X[i-seq_len:i].flatten())
        ys.append(y[i])
    return np.array(Xs), np.array(ys)

def split_data(df):
    y = df['Volatility'].shift(-1).dropna()
    X = df[['LogReturn']].shift(1).dropna()
    common_idx = X.index.intersection(y.index)
    X, y = X.loc[common_idx], y.loc[common_idx]

    n = len(X)
    train_end = int(n * 0.8)
    val_end = int(n * 0.9)

    X_train = X.iloc[:train_end]
    y_train = y.iloc[:train_end]
    X_val = X.iloc[train_end:val_end]
    y_val = y.iloc[train_end:val_end]
    X_test = X.iloc[val_end:]
    y_test = y.iloc[val_end:]

    print(f"\U0001F50D Rozmiar danych dla treningu: {len(X_train)} | walidacja: {len(X_val)} | test: {len(X_test)}")
    return X_train, y_train, X_val, y_val, X_test, y_test

def train_xgb_grid(X_train, y_train, X_val, y_val, X_test, y_test, param_grid, tag, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    pred_dir = os.path.join(output_dir, "wykresy_predykcji")
    model_dir = os.path.join(output_dir, "modele")
    os.makedirs(pred_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    best_models = []

    for n_estimators, max_depth, learning_rate, seq_len, subsample, colsample_bytree, min_child_weight, reg_lambda, reg_alpha in product(
        param_grid['n_estimators'], param_grid['max_depth'], param_grid['learning_rate'], param_grid['sequence_length'],
        param_grid['subsample'], param_grid['colsample_bytree'], param_grid['min_child_weight'], param_grid['reg_lambda'], param_grid['reg_alpha']):

        print(f"\n🚀 Trening: est={n_estimators}, depth={max_depth}, lr={learning_rate}, seq_len={seq_len}, subsample={subsample}, colsample={colsample_bytree}, okres={tag}")

        X_train_seq, y_train_seq = create_features(X_train.values, y_train.values, seq_len)
        X_val_seq, y_val_seq = create_features(X_val.values, y_val.values, seq_len)
        X_test_seq, y_test_seq = create_features(X_test.values, y_test.values, seq_len)

        if len(X_test_seq) == 0:
            print(f" Zbyt mało danych testowych w {tag} dla seq_len={seq_len}, pomijam.")
            continue

        model = XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            min_child_weight=min_child_weight,
            reg_lambda=reg_lambda,
            reg_alpha=reg_alpha,
            objective='reg:squarederror',
            random_state=123
        )

        model.fit(
            np.vstack([X_train_seq, X_val_seq]),
            np.concatenate([y_train_seq, y_val_seq]),
            eval_set=[(X_val_seq, y_val_seq)],
            verbose=False
        )

        y_pred = model.predict(X_test_seq)

        rmse = np.sqrt(mean_squared_error(y_test_seq, y_pred))
        mse = mean_squared_error(y_test_seq, y_pred)
        r2 = r2_score(y_test_seq, y_pred)
        mae = mean_absolute_error(y_test_seq, y_pred)
        mape = np.mean(np.abs((y_test_seq - y_pred) / np.clip(np.abs(y_test_seq), 1e-8, None))) * 100

        print(f" Wyniki: R2={r2:.4f}, RMSE={rmse:.4f},MSE={mse:.4f} ,MAPE={mape:.2f}%")

        result = {
            'okres': tag,
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'sequence_length': seq_len,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'min_child_weight': min_child_weight,
            'reg_lambda': reg_lambda,
            'reg_alpha': reg_alpha,
            'rmse': rmse,
            'mse': mse,
            'mae': mae,
            'r2': r2,
            'mape': mape,
            'model_obj': model
        }
        best_models.append((result, y_test_seq, y_pred))

    df_results = pd.DataFrame([{
        k: v for k, v in r.items() if k != 'model_obj'
    } for r, _, _ in best_models])
    df_results.to_excel(os.path.join(output_dir, f'results_{tag}.xlsx'), index=False)

    for metric in ['r2', 'rmse', 'mse', 'mape']:
        top5 = sorted(best_models, key=lambda x: x[0][metric], reverse=(metric == 'r2'))[:5]
        for i, (res, y_true, y_pred) in enumerate(top5):
            base = f"{tag}_{metric}_top{i+1}_depth{res['max_depth']}_est{res['n_estimators']}_xgb"
            plt.figure(figsize=(10, 4))
            plt.plot(y_true, label='Rzeczywista zmienność')
            plt.plot(y_pred, label='Prognozowana zmienność')
            plt.xticks(ticks=np.arange(len(y_true)), labels=X_test.index[-len(y_true):].strftime('%Y-%m-%d'), rotation=45)
            plt.xlabel("Data")
            plt.ylabel("Zmienność")
            plt.title("Predykcja")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(pred_dir, f"{base}_predykcja.png"))
            plt.close()

            joblib.dump(res['model_obj'], os.path.join(model_dir, f"{base}_model.pkl"))

    return df_results


def run_xgb_for_periods(file_path, param_grid, window_size, output_dir=None):
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    if output_dir is None:
        output_dir = os.path.join("output_xgb", f"{base_name}_xgb", f"window_{window_size}")
    os.makedirs(output_dir, exist_ok=True)

    df = prepare_data_with_sessions(file_path, window_size)
    okresy_df = split_by_periods(df)
    all_results = []

    for okres_name, okres_df in okresy_df.items():
        if len(okres_df) < 50:
            continue
        X_train, y_train, X_val, y_val, X_test, y_test = split_data(okres_df)
        results_df = train_xgb_grid(X_train, y_train, X_val, y_val, X_test, y_test, param_grid, okres_name, output_dir)
        all_results.append(results_df)

    final_df = pd.concat(all_results, ignore_index=True)
    final_df.to_excel(os.path.join(output_dir, 'summary_all_models.xlsx'), index=False)
    return final_df


def process_all_csvs(input_dir, param_grid, window_sizes=[5, 10, 20], output_base_dir="output_xgb"):
    os.makedirs(output_base_dir, exist_ok=True)
    for file in os.listdir(input_dir):
        if file.endswith('.csv'):
            file_path = os.path.join(input_dir, file)
            print(f"\n Przetwarzanie pliku: {file}")
            for window_size in window_sizes:
                print(f"\n Obliczanie dla WINDOW_SIZE={window_size}")
                try:
                    out_dir = os.path.join(output_base_dir, f"{os.path.splitext(file)[0]}_xgb", f"window_{window_size}")
                    results = run_xgb_for_periods(file_path, param_grid, window_size=window_size, output_dir=out_dir)
                    results.to_excel(os.path.join(out_dir, 'summary_all_models.xlsx'), index=False)
                except Exception as e:
                    print(f" Błąd dla {file}, WINDOW_SIZE={window_size}: {e}")


param_grid = {
    'n_estimators': [50, 100, 200, 300],
    'max_depth': [2, 3, 4, 6],
    'learning_rate': [0.01, 0.05, 0.1],
    'sequence_length': [5, 10, 20],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'min_child_weight': [1, 3, 5],
    'reg_lambda': [0.1, 1, 5],
    'reg_alpha': [0, 0.1, 0.5]
}


input_dir = 'dane_wejsciowe'  
process_all_csvs(input_dir, param_grid, window_sizes=[5, 10, 20])
