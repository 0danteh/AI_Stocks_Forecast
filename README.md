# Stock Price Prediction with LSTM

This project demonstrates how to use a Long Short-Term Memory (LSTM) model to predict stock prices. It leverages historical stock price data to train the model, which then makes predictions on future prices. The code is structured to download stock data, prepare the data for training and testing, create and train an LSTM model, make predictions, and evaluate the model's performance.

## Overview

The code is organized into several key sections:

1. **Data Download and Preparation**: The script begins by downloading historical stock price data for a specified ticker symbol using the `yfinance` library. It then prepares the data by calculating the difference in closing prices and splitting the data into training and testing sets.

2. **Model Creation and Training**: An LSTM model is defined and trained using the prepared data. The model architecture includes two LSTM layers, two Dropout layers for regularization, and two Dense layers for output. The model is trained using the Adam optimizer and the mean squared error (MSE) loss function.

3. **Prediction and Evaluation**: After training, the model is used to make predictions on the test set. The predictions are then compared to the actual prices to evaluate the model's performance. The evaluation includes plotting the actual and predicted prices, calculating and plotting the histogram of prediction errors, and calculating the accuracy of the forecast within a specified tolerance.

## Dependencies

- Python 3.x
- TensorFlow
- Keras
- yfinance
- NumPy
- Matplotlib
- Scikit-learn

## How to Use

1. **Install Dependencies**: Ensure you have all the necessary Python libraries installed. You can install them using pip:
```
pip install tensorflow keras yfinance numpy matplotlib scikit-learn
```

2. **Run the Script**: Execute the script in a Python environment. The script will download the stock data, train the LSTM model, make predictions, and display the results.

3. **Customize Parameters**: You can customize the script by changing the ticker symbol, the date range for the data, the number of data points used for training, and the train-test split ratio.

## Code Structure

- **Constants**: Defines constants such as the start and end dates for data download, the ticker symbol, and the train-test split ratio.
- **Functions**: Includes functions for downloading stock data, preparing the train-test split, creating and training the LSTM model, and plotting the results.
- **Main Execution Block**: Contains the main logic of the script, including calling the functions in the correct order.

## Output

The script will output:

- A plot of the actual and predicted prices for the test set.
- A histogram of the prediction errors.
- The accuracy of the forecast within a specified tolerance.
- The mean squared error (MSE) of the predictions.

## Conclusion

This project demonstrates a direct, yet very effective approach to stock price prediction using LSTM models. 
