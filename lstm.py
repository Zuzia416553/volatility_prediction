import pandas as pd
import numpy as np
import os
import pickle
import holidays
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping


def prepare_data_with_sessions_with_window(file_path, window_size):
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


def split_by_periods(df):
    okresy = {
        
        '2013-2015_regulacje': ('2013-09-01', '2015-10-31'),
       '2015-2017_podatek_frankowicze': ('2015-11-01', '2017-12-31'),
        '2018-2019_przed_covid': ('2018-01-01', '2019-12-31'),
       '2020-2021_covid': ('2020-01-01', '2021-12-31'),
        '2022-2023_wojna_inflacja': ('2022-01-01', '2023-12-31'),
    }
    return {nazwa: df.loc[start:end].copy() for nazwa, (start, end) in okresy.items() if not df.loc[start:end].empty}

def split_data(df):
    y = df['Volatility'].shift(-1).dropna()
    X = df[['LogReturn']].shift(1).dropna()
    common_idx = X.index.intersection(y.index)
    X, y = X.loc[common_idx], y.loc[common_idx]

    n = len(X)
    train_end = int(n * 0.8)
    val_end = int(n * 0.9)

    return X.iloc[:train_end], y.iloc[:train_end], X.iloc[train_end:val_end], y.iloc[train_end:val_end], X.iloc[val_end:], y.iloc[val_end:]


def create_sequences(X, y, seq_length):
    Xs, ys = [], []
    for i in range(seq_length, len(X)):
        Xs.append(X[i-seq_length:i])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)

def train_lstm_grid(X_train, y_train, X_val, y_val, X_test, y_test, param_grid, tag, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    loss_dir = os.path.join(output_dir, "wykresy_loss")
    pred_dir = os.path.join(output_dir, "wykresy_predykcji")
    modele_dir = os.path.join(output_dir, "modele")
    os.makedirs(loss_dir, exist_ok=True)
    os.makedirs(pred_dir, exist_ok=True)
    os.makedirs(modele_dir, exist_ok=True)

    best_models = []

    for units in param_grid['units']:
        for epochs in param_grid['epochs']:
            for batch_size in param_grid['batch_size']:
                for seq_len in param_grid['sequence_length']:
                    for num_layers in param_grid['num_layers']:
                        for scaling in param_grid['scaling']:
                            for learning_rate in param_grid['learning_rate']:
                                for dropout_rate in param_grid['dropout_rate']:

                                    print(f"\n Trening: units={units}, layers={num_layers}, seq_len={seq_len}, ep={epochs}, bs={batch_size}, lr={learning_rate}, dropout={dropout_rate}, okres={tag}")

                                    scaler_X = MinMaxScaler()
                                    scaler_y = MinMaxScaler()

                                    X_train_scaled = scaler_X.fit_transform(X_train)
                                    X_val_scaled = scaler_X.transform(X_val)
                                    X_test_scaled = scaler_X.transform(X_test)

                                    y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1))
                                    y_val_scaled = scaler_y.transform(y_val.values.reshape(-1, 1))

                                    X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_scaled, seq_len)
                                    X_val_seq, y_val_seq = create_sequences(X_val_scaled, y_val_scaled, seq_len)
                                    X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test.values.reshape(-1, 1), seq_len)

                                    model = Sequential()
                                    for layer_idx in range(num_layers):
                                        neurons = units if scaling == 'constant' else units // (2 ** layer_idx)
                                        return_seq = layer_idx < num_layers - 1
                                        if layer_idx == 0:
                                            model.add(LSTM(neurons, activation='relu', return_sequences=return_seq, input_shape=(seq_len, X_train_seq.shape[2])))
                                        else:
                                            model.add(LSTM(neurons, activation='relu', return_sequences=return_seq))
                                        model.add(Dropout(dropout_rate))
                                    model.add(Dense(1))

                                    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')

                                    early_stop = EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True)

                                    history = model.fit(
                                        X_train_seq, y_train_seq,
                                        epochs=epochs,
                                        batch_size=batch_size,
                                        validation_data=(X_val_seq, y_val_seq),
                                        verbose=1,
                                        shuffle=False,
                                        callbacks=[early_stop]
                                    )

                                    y_pred_scaled = model.predict(X_test_seq)
                                    y_pred = scaler_y.inverse_transform(y_pred_scaled).flatten()
                                    y_test_seq = y_test_seq.flatten()

                                    rmse = np.sqrt(mean_squared_error(y_test_seq, y_pred))
                                    mse = mean_squared_error(y_test_seq, y_pred)
                                    r2 = r2_score(y_test_seq, y_pred)
                                    mape = np.mean(np.abs((y_test_seq - y_pred) / y_test_seq)) * 100
                                    mae = np.mean(np.abs(y_test_seq - y_pred))

                                    print(f" R2={r2:.4f}, RMSE={rmse:.6f}, MSE={mse:.6f}, MAPE={mape:.2f}%")

                                    final_epoch = len(history.history['loss'])
                                    early_stopping_used = final_epoch < epochs

                                    result = {
                                        'okres': tag,
                                        'units': units,
                                        'epochs_requested': epochs,
                                        'epochs_completed': final_epoch,
                                        'early_stopping': early_stopping_used,
                                        'batch_size': batch_size,
                                        'seq_len': seq_len,
                                        'num_layers': num_layers,
                                        'scaling': scaling,
                                        'learning_rate': learning_rate,
                                        'dropout_rate': dropout_rate,
                                        'rmse': rmse,
                                        'mse': mse,
                                        'mae' : mae,
                                        'r2': r2,
                                        'mape': mape,
                                    }


                                    best_models.append((result, history, y_test_seq, y_pred))

    df_results = pd.DataFrame([r for r, _, _, _ in best_models])
    df_results.to_excel(os.path.join(output_dir, f'results_{tag}.xlsx'), index=False)

    for metric in ['r2', 'rmse', 'mse', 'mape']:
        top5 = sorted(best_models, key=lambda x: x[0][metric], reverse=(metric == 'r2'))[:5]
        for i, (res, hist, y_true, y_pred) in enumerate(top5):
            base = f"{tag}_{metric}_top{i+1}_units{res['units']}_ep{res['epochs_completed']}_bs{res['batch_size']}_lstm"


            with open(os.path.join(modele_dir, f"{base}.pkl"), "wb") as f:
                pickle.dump({
                    "model": hist.model,
                    "scaler_X": scaler_X,
                    "scaler_y": scaler_y
                }, f)

            
            plt.figure(figsize=(10, 4))
            plt.plot(y_test.index[-len(y_true):], y_true, label='Rzeczywista zmienność')
            plt.plot(y_test.index[-len(y_pred):], y_pred, label='Prognozowana zmienność')
            plt.title("Predykcja")
            plt.xlabel("Data")
            plt.ylabel("Zmienność")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(pred_dir, f"{base}_predykcja.png"))
            plt.close()

            # Wykres strat
            plt.figure(figsize=(10, 4))
            plt.plot(hist.history['loss'], label='Błąd trenowania')
            plt.plot(hist.history['val_loss'], label='Błąd walidacyjny')
            plt.title("Wykres strat")
            plt.xlabel('Epoka')
            plt.ylabel('Błąd')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(loss_dir, f"{base}_loss.png"))
            plt.close()

    return df_results


def run_lstm_for_all_files(input_dir, param_grid, window_sizes):
    csv_files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]

    for file in csv_files:
        file_path = os.path.join(input_dir, file)
        base_name = os.path.splitext(file)[0]
        base_output_dir = os.path.join("lstm_wyniki", f"{base_name}_lstm")

        for window_size in window_sizes:
            print(f"\n Przetwarzanie pliku: {file} | WINDOW_SIZE={window_size}")
            output_dir = os.path.join(base_output_dir, f"window_{window_size}")
            os.makedirs(output_dir, exist_ok=True)

            df = prepare_data_with_sessions_with_window(file_path, window_size)
            okresy_df = split_by_periods(df)
            all_results = []

            for okres_name, okres_df in okresy_df.items():
                if len(okres_df) < 50:
                    continue

                okres_output_dir = os.path.join(output_dir, okres_name)
                os.makedirs(okres_output_dir, exist_ok=True)

                X_train, y_train, X_val, y_val, X_test, y_test = split_data(okres_df)
                results_df = train_lstm_grid(X_train, y_train, X_val, y_val, X_test, y_test, param_grid, okres_name, okres_output_dir)
                all_results.append(results_df)

            if all_results:
                final_df = pd.concat(all_results, ignore_index=True)
                final_df.to_excel(os.path.join(output_dir, 'summary_all_models.xlsx'), index=False)


param_grid = {
   'units': [128],        
    'epochs': [20, 40, 60, 80],        
    'batch_size': [8, 16],
    'sequence_length': [5, 10, 20],
    'num_layers': [1, 2, 3],
    'scaling': ['constant'],
    'learning_rate': [0.001],
    'dropout_rate': [0, 0.2]
}

run_lstm_for_all_files('dane_wejsciowe', param_grid, window_sizes=[5, 10, 20])
