import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dropout, Dense, Activation
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers
import yfinance as yf
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# Constants
START_DATE = datetime(2010, 1, 1)
END_DATE = datetime(2020, 1, 1)
TICKER_SYMBOL = "AAPL"
TRAIN_SPLIT = 0.7
DATA_SET_POINTS = 21

def download_stock_data(ticker_symbol, start_date, end_date):
    stock_df = yf.download(tickers=ticker_symbol, start=start_date, end=end_date)
    new_df = stock_df[['Adj Close']].copy()
    return new_df

def prepare_train_test_split(new_df, data_set_points, train_split):
    new_df.reset_index(inplace=True)
    new_df.drop(0, inplace=True)
    split_index = int(len(new_df) * train_split)
    train_data = new_df[:split_index]
    test_data = new_df[split_index:].reset_index(drop=True)
    train_diff = train_data['Adj Close'].diff().dropna().values
    test_diff = test_data['Adj Close'].diff().dropna().values
    X_train = np.array([train_diff[i : i + data_set_points] for i in range(len(train_diff) - data_set_points)])
    y_train = np.array([train_diff[i + data_set_points] for i in range(len(train_diff) - data_set_points)])
    y_valid = train_data['Adj Close'].tail(len(y_train) // 10).values
    y_valid = y_valid.reshape(-1, 1)
    X_test = np.array([test_diff[i : i + data_set_points] for i in range(len(test_diff) - data_set_points)])
    y_test = test_data['Adj Close'].shift(-data_set_points).dropna().values
    return X_train, y_train, X_test, y_test, test_data

def create_lstm_model(X_train, y_train, data_set_points):
    tf.random.set_seed(20)
    np.random.seed(10)
    lstm_input = Input(shape=(data_set_points, 1), name='lstm_input')
    inputs = LSTM(21, name='lstm_0', return_sequences=True)(lstm_input)
    inputs = Dropout(0.1, name='dropout_0')(inputs)
    inputs = LSTM(32, name='lstm_1')(inputs)
    inputs = Dropout(0.05, name='dropout_1')(inputs)
    inputs = Dense(32, name='dense_0')(inputs)
    inputs = Dense(1, name='dense_1')(inputs)
    output = Activation('linear', name='output')(inputs)
    model = Model(inputs=lstm_input, outputs=output)
    adam = optimizers.Adam(lr=0.002)
    model.compile(optimizer=adam, loss='mse')
    model.fit(x=X_train, y=y_train, batch_size=15, epochs=25, shuffle=True, validation_split=0.1)
    return model

def plot_predictions(actual1, data, data_set_points):
    plt.gcf().set_size_inches(12, 8, forward=True)
    plt.title('Plot of real price and predicted price against number of days for test set')
    plt.xlabel('Number of days')
    plt.ylabel('Adjusted Close Price($)')
    plt.plot(actual1[1:], label='Actual Price')
    plt.plot(data, label='Predicted Price')
    plt.legend(['Actual Price', 'Predicted Price'])
    plt.show()

def plot_error_histogram(error):
    plt.hist(error, bins=25)
    plt.xlabel('Prediction Error ($)')
    plt.title('Histogram of prediction errors')
    plt.ylabel('Frequency')
    plt.show()

if __name__ == "__main__":
    stock_df = download_stock_data(TICKER_SYMBOL, START_DATE, END_DATE)
    
    X_train, y_train, X_test, y_test, test_data = prepare_train_test_split(stock_df, DATA_SET_POINTS, TRAIN_SPLIT)
    
    model = create_lstm_model(X_train, y_train, DATA_SET_POINTS)
    
    y_pred = model.predict(X_test)
    y_pred = y_pred.flatten()
    
    actual1 = np.array([test_data['Adj Close'][i + DATA_SET_POINTS] for i in range(len(test_data) - DATA_SET_POINTS)])
    actual2 = actual1[:-1]
    data = np.add(actual2, y_pred)
    
    plot_predictions(actual1, data, DATA_SET_POINTS)
    
    error = actual1[1:] - data
    plot_error_histogram(error)
    
    percentage_tolerance = 0.10
    diff = np.abs(actual1[1:] - data)
    within_tolerance = np.where((diff / actual1[1:]) <= percentage_tolerance, 1, 0)
    accuracy = np.sum(within_tolerance) / len(within_tolerance) * 100
    print(f"The Forecast is Accurate by {accuracy}%")
    
    mse = mean_squared_error(actual1[1:], data, squared=False)
    print(f"The Mean Squared Error (MSE) is: {mse}")
