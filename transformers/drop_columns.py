from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np


class DropColumns(BaseEstimator, TransformerMixin):

    def __init__(self, columns=None):
        self.columns = columns

    def fit(self, X, y=None):
        if self.columns is None:
            self.drop_columns_ = np.array(['UDI', 'Product ID', 'Torque [Nm]', 'TWF', 'PWF', 'OSF'])
        else:
            self.drop_columns_ = self.columns
        return self

    def transform(self, X, y=None):
        return X.drop(columns=self.drop_columns_)