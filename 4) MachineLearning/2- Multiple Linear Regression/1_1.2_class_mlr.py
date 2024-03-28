import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score


class MultipleLinearRegression:
    def __init__(self, file_path):
        self.df = pd.read_csv(file_path)
        self.split_prepare_data()

    def split_prepare_data(self):
        self.X = self.df.drop("sales", axis=1)
        self.y = self.df[["sales"]]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.20, random_state=1
        )
        return self.X_train, self.X_test, self.y_train, self.y_test
        
    def fit_model(self):
        self.reg_model = LinearRegression().fit(self.X_train, self.y_train)

    def predict_train_test(self):
        self.y_pred_train = self.reg_model.predict(self.X_train)
        self.y_pred_test = self.reg_model.predict(self.X_test)
        return self.y_pred_train, self.y_pred_test

    def evaluate_model(self):
        metrics_train = self.calculate_metrics(self.y_train, self.y_pred_train)
        metrics_test = self.calculate_metrics(self.y_test, self.y_pred_test)
        return {"train": metrics_train, "test": metrics_test}

    def calculate_metrics(self, true_values, predicted_values):
        mse = mean_squared_error(true_values, predicted_values)
        rmse = np.sqrt(mse)
        rsquare = self.reg_model.score(self.X_train, self.y_train)
        return {"MSE": mse, "RMSE": rmse, "R-Square": rsquare}

    def cv(self, n=10):
        cv_mean = np.mean(
            np.sqrt(
                -cross_val_score(
                    self.reg_model,
                    self.X_train,
                    self.y_train,
                    cv=n,
                    scoring="neg_mean_squared_error",
                )
            )
        )
        print(f"Cross-validation RMSE: {cv_mean}")
        return cv_mean

    def visualize_model(self):
        sets = [(self.y_train, self.y_pred_train, 'Training'), (self.y_test, self.y_pred_test, 'Test')]
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))

        for ax, (y_true, y_pred, dataset_type) in zip(axes, sets):
            ax.scatter(y_true, y_pred, label=f'Actual vs Predicted ({dataset_type})', color='blue')
            ax.plot(y_true, y_true, color='red', linewidth=2, label='Perfect Prediction')
            ax.set_title(f'Actual vs Predicted Values ({dataset_type} Set)')
            ax.set_xlabel('Actual Values')
            ax.set_ylabel('Predicted Values')
            ax.legend()

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    mlr = MultipleLinearRegression("datasets/advertising.csv")
    mlr.fit_model()
    mlr.predict_train_test()
    evaluation_results = mlr.evaluate_model()
    print("Evaluation results:")

    for dataset, metrics in evaluation_results.items():
        print(f"{dataset.capitalize()} set:")
        for metric, value in metrics.items():
            print(f"{metric}: {value}")
    cv_result = mlr.cv()

    mlr.visualize_model()