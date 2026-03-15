# main.py - Utility functions for regression modeling

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from seaborn import color_palette
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import mean_absolute_percentage_error as MAPE
from sklearn.model_selection import train_test_split as tts
from sklearn.linear_model import LinearRegression
from IPython.display import display, HTML

def regression_error_metrics(y, yhat):
    """Calculate and display regression error metrics."""
    from sklearn.metrics import mean_absolute_error as MAE, mean_squared_error as MSE, mean_absolute_percentage_error as MAPE
    from IPython.display import display, HTML
    ME = np.round(np.mean(y - yhat), 3)
    MPE = np.round(np.mean((y - yhat) / y), 3)
    myMAE = np.round(MAE(y, yhat), 3)
    myMSE = np.round(MSE(y, yhat), 3)
    myMAPE = np.round(MAPE(y, yhat), 3)
    data = [[ME, MPE, myMAE, myMAPE, myMSE]]
    df = pd.DataFrame(data, columns=["ME", "MPE", "MAE", "MAPE", "MSE"])
    display(HTML(df.to_html(index=False)))


def generate_predictions_and_save(model, test_df, feature_cols, output_filename):
    """Generate predictions, set negatives to zero, and save to CSV."""
    predictions = model.predict(test_df[feature_cols])
    predictions = np.where(predictions < 0, 0, predictions)
    output = pd.DataFrame({"id": test_df["id"], 'Rings': predictions})
    output.to_csv(output_filename, index=False)

def plot_regression_diagnostics(y_true, y_pred, model_name="Model"):
    """Plot Q-Q and fitted vs residuals plots."""
    import matplotlib.pyplot as plt
    from scipy import stats
    residuals = y_true - y_pred
    plt.figure(figsize=(12, 5))
    # Q-Q plot
    plt.subplot(1, 2, 1)
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title(f"Q-Q Plot of Residuals for {model_name}", fontsize=16)
    plt.xlabel("Theoretical Quantiles", fontsize=12)
    plt.ylabel("Ordered Residuals", fontsize=12)
    plt.grid()
    # Fitted vs residuals
    plt.subplot(1, 2, 2)
    plt.scatter(y_pred, residuals)
    plt.axhline(0, color='r', linestyle='--')
    plt.title(f"Fitted vs Residuals for {model_name}", fontsize=16)
    plt.xlabel("Fitted Values", fontsize=12)
    plt.ylabel("Residuals", fontsize=12)
    plt.grid()
    plt.tight_layout()
    plt.show()

