import joblib


class Predict:

    def __init__(self):
        self.lin_reg = joblib.load("models/AI4I_lin_reg.pkl")
        self.ridge = joblib.load("models/AI4I_ridge.pkl")
        self.lasso = joblib.load("models/AI4I_lasso.pkl")
        self.elastic_net = joblib.load("models/AI4I_elastic_net.pkl")

    def lin_reg_predict(self, X):
        return self.lin_reg.predict(X)[0]

    def ridge_predict(self, X):
        return self.ridge.predict(X)[0]

    def lasso_predict(self, X):
        return self.lasso.predict(X)[0]

    def elastic_net_predict(self, X):
        return self.elastic_net.predict(X)[0]
