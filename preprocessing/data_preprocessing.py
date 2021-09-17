import urllib
import pandas as pd
from pipelines.custom_pipelines import CustomPipeline


class Preprocessing:

    def __init__(self):
        file = urllib.request.urlretrieve(
            "https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv")
        df = pd.read_csv(file[0])
        self.X = df.drop(columns=['Air temperature [K]'])
        self.y = df['Air temperature [K]'].values
        self.scaler = CustomPipeline().total_transform_pipeline()
        self.X = self.scaler.fit_transform(self.X, self.y)

    def get_processed_data(self, X):
        return self.scaler.transform(X)