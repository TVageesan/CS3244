import matplotlib.pyplot as plt
import seaborn as sns

def plot_prediction(y_test, y_pred):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--', color='red')
    plt.xlabel('Actual Resale Price')
    plt.ylabel('Predicted Resale Price')
    plt.title('Actual vs Predicted Resale Price')
    plt.grid(True)
    plt.show()

def plot_residuals(y_test, y_pred):
    residuals = y_test - y_pred

    plt.figure(figsize=(8, 5))
    sns.histplot(residuals, kde=True, bins=50)
    plt.title('Distribution of Residuals')
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show() 