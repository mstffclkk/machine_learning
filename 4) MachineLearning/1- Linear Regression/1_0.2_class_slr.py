import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

class SimpleLinearRegression:
    def __init__(self,file_path):
        self.df = pd.read_csv(file_path)
        self.prepare_data()

    def prepare_data(self):
        self.X = self.df[["TV"]]
        self.y = self.df[["sales"]]

    def visualize_data(self):
        sns.jointplot(x="TV", y="sales",data=self.df, kind="reg")
        plt.show()
    
    def fit_model(self):
        self.reg_model = LinearRegression().fit(self.X, self.y)

    def predict(self):
        return self.reg_model.predict(self.X)

    def visualize_model(self):
        g = sns.regplot(x=self.X, y=self.y, scatter_kws={'color':'b','s':9}, ci=False, color='r')
        g.set_title(self.get_model_equation())
        g.set_ylabel("Satış Sayısı")
        g.set_xlabel("TV Harcamaları")
        plt.xlim(-10, 310)
        plt.ylim(bottom=0)
        plt.show()    

    def get_model_equation(self):
        return f"Model Denklemi: Sales = {round(self.reg_model.intercept_[0],2)} + TV*{round(self.reg_model.coef_[0][0], 2)}"

    def evaluate_model(self):
        y_pred = self.reg_model.predict(self.X)
        mse = mean_squared_error(self.y, y_pred, squared=True)
        rmse = mean_squared_error(self.y, y_pred, squared=False)
        rsquare = self.reg_model.score(self.X, self.y)
        print(f"MSE: {mse}",
              f"RMSE: {rmse}",
              f"R-Square: {rsquare}")

lr = SimpleLinearRegression("datasets/advertising.csv")
lr.visualize_data()
lr.fit_model()
lr.predict()
lr.visualize_model()
lr.evaluate_model()

