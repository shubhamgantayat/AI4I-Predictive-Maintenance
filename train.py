import os

from sklearn.linear_model import ElasticNet
import mlflow
import numpy as np
import argparse
import joblib
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from preprocessing.data_preprocessing import Preprocessing


def main(alpha):
    # enable autologging
    mlflow.sklearn.autolog()
    preprocess = Preprocessing()
    X_train, X_test, y_train, y_test = train_test_split(preprocess.X, preprocess.y, test_size=0.2, random_state=101)
    model = ElasticNet(alpha=alpha)
    with mlflow.start_run() as run:
        model.fit(X_train, y_train)
        if "AI4I_elastic_net.pkl" in os.listdir('models/'):
            prev_model = joblib.load("models/AI4I_elastic_net.pkl")
            pred = model.predict(X_test)
            prev_pred = prev_model.predict(X_test)
            error = mean_squared_error(y_test, pred)
            prev_error = mean_squared_error(y_test, prev_pred)
            if prev_error > error:
                joblib.dump(model, "models/AI4I_elastic_net.pkl")
        else:
            joblib.dump(model, "models/AI4I_elastic_net.pkl")
        metrics = mlflow.sklearn.eval_and_log_metrics(model, X_test, y_test, prefix="val_")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--alpha", help="alpha value")
    args = parser.parse_args()
    if args.alpha:
        alpha = float(args.alpha)
    else:
        alpha = 1.0
    main(alpha)
