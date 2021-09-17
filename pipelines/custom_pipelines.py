from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from transformers.drop_columns import DropColumns


class CustomPipeline:

    @staticmethod
    def column_transform_pipeline():
        num_attribs = ['Process temperature [K]', 'Rotational speed [rpm]', 'Tool wear [min]']
        cat_attribs = ['Type']

        column_transformer = ColumnTransformer([
            ("num", StandardScaler(), num_attribs),
            ("cat", OneHotEncoder(), cat_attribs)
        ])
        return column_transformer

    @staticmethod
    def total_transform_pipeline():
        total_transformer = Pipeline([
            ("drop_columns", DropColumns()),
            ("standardization", CustomPipeline().column_transform_pipeline())
        ])
        return total_transformer
