# Statistical vs. Machine Learning Methods for Financial Forecasting

A Python benchmark of statistical and machine learning methods for predicting **next-day SPY ETF log returns**. The study emphasizes chronological validation, reproducible model comparisons, and the distinction between model complexity and predictive accuracy.

**[Read the full analysis (Jupyter notebook)](https://nbviewer.org/github/minjee-kim/financial-forecasting/blob/main/analysis.ipynb)**

## Research question

Do nonlinear machine learning methods improve out-of-sample numerical return forecasts relative to classical statistical benchmarks?

## Data and design

- **Asset:** SPDR S&P 500 ETF Trust (SPY); adjusted daily prices and volume, January 2010–September 2026.
- **Target:** next-day log return, $r_{t+1}=\log(P_{t+1}/P_t)$.
- **Predictors:** lagged returns, rolling volatility, momentum, volume indicators, moving-average deviation, and intraday price measures (14 candidate features).
- **Evaluation:** chronological 60% training / 20% validation / 20% test split; five-fold expanding-window cross-validation for hyperparameter tuning, with a one-observation gap.

## Forecasting models

| Family | Models |
| --- | --- |
| Unconditional statistical benchmarks | Historical mean; Student's $t$ location; Cauchy location |
| Rolling statistical benchmarks | 252-day rolling mean; rolling Student's $t$ location; rolling Cauchy location |
| Linear regression | Ordinary least squares (OLS), Ridge, LASSO |
| Tree ensembles | Random forest, gradient boosting |

The statistical location benchmarks produce point forecasts; the rolling versions re-estimate their parameters daily. Regression and tree-ensemble models are fitted on the training sample and held fixed over the validation period.

## Validation findings

**Validation period: January 29, 2020–May 25, 2023 (838 observations).** Results below are *validation*, not final held-out test, performance.

| Metric | Best model | Result |
| --- | --- | ---: |
| RMSE | Ridge | **0.015338** |
| MAE | Rolling Student's $t$ | **0.010244** |
| Directional accuracy | Random forest | **54.65%** |

Ridge achieved a marginal improvement over the historical-mean RMSE of **0.015346**, corresponding to $R^2_{\mathrm{OOS}}=0.0010$ (approximately **0.1%** lower squared prediction error). Rolling Student's $t$ had the lowest MAE but did not improve RMSE over the historical mean. Neither random forest nor gradient boosting improved RMSE relative to that baseline.

**Takeaway:** Across these specifications, greater model flexibility did not yield a meaningful improvement in numerical next-day return forecasts during validation.

## Limitations and status

The analysis uses SPY-derived price and volume predictors only. Most predictive models remain fixed during validation, while rolling statistical benchmarks update daily. Performance differences are small and have not been tested for statistical significance. The **May 2023–September 2026 test period remains unevaluated**; the findings above should not be read as final test results.

## Implementation

The [analysis notebook](analysis.ipynb) documents data exploration, feature construction, training/validation design, predictions, and evaluation. Reusable model implementations are organized under `src/models/`.

**Tools:** Python, pandas, NumPy, SciPy, scikit-learn, matplotlib, yfinance.
