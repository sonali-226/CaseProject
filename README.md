
# Stock Price Prediction Using LSTM

This document provides a comprehensive overview of the stock price prediction project utilizing an LSTM (Long Short-Term Memory) model. The objective of this project is to predict stock prices for companies like Microsoft (MSFT) and Apple (AAPL) based on historical price data. We aim to build a model that can predict future stock prices and analyze the accuracy of the predictions through various performance metrics.

## Table of Contents

1. [Introduction](#introduction)
2. [Requirements and Updates](#requirements-and-updates)
3. [Roles and Responsibilities](#roles-and-responsibilities)
4. [Supporting Documentation](#supporting-documentation)
    - [Process Documentation](#process-documentation)
    - [Conceptual Documentation](#conceptual-documentation)
5. [Conclusion](#conclusion)

---

## 1. Introduction

This document outlines the steps involved in building and deploying a stock price prediction model using LSTM (Long Short-Term Memory) neural networks. The model is trained using historical stock prices of Microsoft (MSFT) and Apple (AAPL) and is evaluated on a test dataset to predict future stock prices for a specified period (e.g., March 2024).

### Project Overview:

- **Objective**: Predict stock prices for Microsoft and Apple using historical data and LSTM neural networks.
- **Tech Stack**: Python, LSTM, Keras, Pandas, Numpy, Matplotlib, yFinance, Scikit-learn.
- **Scope**: Predict stock prices for a specified period (e.g., March 2024) based on past data from Yahoo Finance.

---

## 2. Requirements and Updates

### 2.1 Data Collection and Preparation

- **Requirement**: Download historical stock price data for Microsoft and Apple from Yahoo Finance using the `yfinance` library.
- **Update**: The stock data is fetched for both the training and testing periods, with a split between past data for training and recent data for testing the model's prediction.

### 2.2 Data Preprocessing

- **Requirement**: Normalize the data using `MinMaxScaler` to scale the stock prices between 0 and 1.
- **Update**: Data is preprocessed to ensure that the model can efficiently learn from it. A sliding window technique is applied to generate features and labels for training.

### 2.3 Model Development

- **Requirement**: Implement an LSTM model using Keras.
- **Update**: A two-layer LSTM model is built and trained to predict the stock price based on past prices. The model is compiled with the Adam optimizer and mean squared error as the loss function.

### 2.4 Model Training

- **Requirement**: Train the LSTM model on the prepared dataset.
- **Update**: The model is trained on historical data from January 2020 to March 2024, with the training data split into smaller batches for efficient learning.

### 2.5 Evaluation

- **Requirement**: Evaluate the performance of the model using performance metrics like Mean Squared Error (MSE) and Mean Absolute Percentage Error (MAPE).
- **Update**: MSE and MAPE are calculated for the test set to assess the accuracy of the predictions.

### 2.6 Prediction and Visualization

- **Requirement**: Generate stock price predictions for the test period and visualize the results.
- **Update**: The predictions for March 2024 are compared to the actual prices, and the results are visualized using Matplotlib.

---

## 3. Supporting Documentation

### 3.1 Process Documentation

#### Data Collection Process

The steps for collecting historical stock price data using the `yfinance` library:

1. **Install the Required Libraries**:

   ```bash
   pip install yfinance
   ```

2. **Import the Library**:

   ```python
   import yfinance as yf
   import pandas as pd
   ```

3. **Download Stock Data**:

   ```python
   msft = yf.download("MSFT", start="2020-01-01", end="2024-03-01")
   aapl = yf.download("AAPL", start="2020-01-01", end="2024-03-01")
   ```

4. **Save the Data**:

   ```python
   msft.to_csv('msft_data.csv')
   aapl.to_csv('aapl_data.csv')
   ```

5. **Verify the Data**:

   ```python
   msft_data = pd.read_csv('msft_data.csv')
   print(msft_data.head())
   ```

#### Data Preprocessing Flow

1. **Cleaning the Data**:

   ```python
   msft_data = msft_data.dropna()
   ```

2. **Feature Engineering**:

   ```python
   data = msft_data[['Close']]
   ```

3. **Normalization**:

   ```python
   from sklearn.preprocessing import MinMaxScaler
   scaler = MinMaxScaler(feature_range=(0, 1))
   scaled_data = scaler.fit_transform(data)
   ```

4. **Splitting Data**:

   ```python
   train_size = int(len(scaled_data) * 0.8)
   train_data = scaled_data[:train_size]
   test_data = scaled_data[train_size:]
   ```

5. **Create Time Series Data**:

   ```python
   def create_dataset(data, window_size):
       X, y = [], []
       for i in range(len(data) - window_size - 1):
           X.append(data[i:(i + window_size), 0])
           y.append(data[i + window_size, 0])
       return np.array(X), np.array(y)

   window_size = 60
   X_train, y_train = create_dataset(train_data, window_size)
   X_test, y_test = create_dataset(test_data, window_size)
   ```

### 3.2 Conceptual Documentation

#### LSTM Explanation

Long Short-Term Memory (LSTM) is a type of Recurrent Neural Network (RNN) designed to handle sequences of data. LSTM networks are particularly useful for time-series prediction because they can learn from long-term dependencies in sequential data, unlike traditional feed-forward networks.

- **Architecture**: LSTM consists of memory cells that retain information over time, allowing the model to learn long-term dependencies. These cells have gates: input gate, forget gate, and output gate, which control the flow of information.
- **Suitability for Time-Series**: In stock price prediction, the prices are sequential in nature (i.e., past prices influence future prices), making LSTM a perfect choice for this task.

#### Stock Market Predictive Modeling

Predictive modeling involves building statistical or machine learning models to predict future values based on historical data. In stock price prediction, models like LSTM are trained on historical price data, considering factors such as market trends, economic events, and company-specific developments.

- **Data for Stock Price Prediction**:
  - Historical stock prices
  - Technical indicators (e.g., Moving Averages, RSI)
  - Sentiment analysis (e.g., social media or news data)

- **Types of Models**:
  - Time-series models (e.g., ARIMA, LSTM)
  - Regression models (e.g., Linear Regression, Support Vector Machines)
  - Ensemble methods (e.g., Random Forest, Gradient Boosting)

#### Model Optimization

To improve the performance of the model, several optimization techniques can be applied:

1. **Hyperparameter Tuning**: Tuning hyperparameters like the number of LSTM units, batch size, and the number of layers can help improve the model’s performance. This can be done through techniques such as Grid Search or Random Search.
2. **Regularization**: Techniques like Dropout or L2 regularization can prevent the model from overfitting and improve generalization.
3. **Learning Rate Scheduling**: Adjusting the learning rate during training can help the model converge faster and avoid overshooting the optimal weights.
4. **Ensemble Models**: Combining multiple models (e.g., LSTM, Random Forest, and XGBoost) in an ensemble method can improve prediction accuracy by leveraging the strengths of different models.

---

## 4. Conclusion

This documentation provides an in-depth overview of the stock price prediction project using LSTM. It covers all aspects from data collection to model evaluation. Additionally, it includes the supporting documentation required to ensure the success of the project, from process guides to conceptual explanations.
