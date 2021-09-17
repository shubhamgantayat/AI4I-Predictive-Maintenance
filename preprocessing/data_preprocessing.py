import urllib
import pandas as pd
from pipelines.custom_pipelines import CustomPipeline


class Preprocessing:

    def __init__(self):
        file = urllib.request.urlretrieve(
            "https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv")
        df = pd.read_csv(file[0])
        X = df.drop(columns=['Air temperature [K]'])
        y = df['Air temperature [K]'].values
        self.scaler = CustomPipeline().total_transform_pipeline()
        self.scaler.fit(X, y)

    def get_processed_data(self, X):
        return self.scaler.transform(X)