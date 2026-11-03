# Statistical vs. Machine Learning Methods for Financial Forecasting

This project investigates whether machine learning methods improve out-of-sample predictions of daily SPY ETF returns relative to classical statistical models.

The analysis focuses on predictive performance, temporal validation, and the tradeoff between model complexity and generalization.


### 1. Research Question
Do flexible machine learning models improve out-of-sample forecasts relative to linear models and a historical-mean benchmark? 

### 2. Data and Experimental Design
The study uses historical daily SPY ETF price and volume data.

The prediction target is the next-day log return:

$$
r_{t+1}=\log(P_{t+1}/P_t).
$$

Predictors include lagged returns, rolling volatility, momentum, volume changes, and moving-average deviations.

### 3. Models 
Six forecasting methods are compared:
* Historical mean (benchmark)
* Ordinary least squares (OLS)
* Ridge regression
* Lasso regression
* Random forest regression
* Gradient boosting regression

Models are evaluated using chronological data splits and time-series cross-validation. 
Preprocessing and hyperparameter tuning are restricted to the appropriate training periods to prevent bias. 

Out-of-sample forecasts are compared using:
* Root mean squared error (RMSE)
* Mean absolute error (MAE)
* Out-of-sample (R^2) relative to the historical-mean benchmark
* Directional accuracy

### 4. Results 
To be completed after model estimation.

### 5. Discussion and Limitations
To be completed after evaluating the out-of-sample results.

### 6. Reproducibility
Installation instructions and experiment execution commands will be added as the implementation is completed.

### 7. Tools
Python, pandas, NumPy, scikit-learn, matplotlib, and yfinance.
