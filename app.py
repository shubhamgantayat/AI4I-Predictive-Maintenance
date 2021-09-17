from flask import Flask, request, render_template
from flask_cors import CORS
import pandas as pd
from preprocessing.data_preprocessing import Preprocessing
from prediction.model_prediction import Predict

app = Flask(__name__)
preprocessing = Preprocessing()
predict = Predict()
CORS(app)


@app.route('/')
def home_page():
    return render_template("form.html")


@app.route('/profile-report')
def profile_report():
    return render_template("my_data.html")


@app.route('/predict', methods=['POST'])
def predict_linear():
    try:
        if request.method == 'POST':
            columns = ['UDI', 'Product ID', 'Type', 'Process temperature [K]', 'Rotational speed [rpm]',
                       'Torque [Nm]', 'Tool wear [min]', 'Machine failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']
            str_attributes = ["Product ID", "Type"]
            float_attributes = ['Process temperature [K]', 'Torque [Nm]']
            records = dict()
            for col in columns:
                if col in str_attributes:
                    records[col] = str(request.form[col])
                elif col in float_attributes:
                    records[col] = float(request.form[col])
                else:
                    records[col] = int(request.form[col])
            X = pd.DataFrame.from_dict([records])
            X_tf = preprocessing.get_processed_data(X)
            if request.form['Model'] == 'lin_reg':
                result = predict.lin_reg_predict(X_tf)
            elif request.form['Model'] == 'ridge':
                result = predict.ridge_predict(X_tf)
            elif request.form['Model'] == 'lasso':
                result = predict.lasso_predict(X_tf)
            else:
                result = predict.elastic_net_predict(X_tf)
            result = "Air temperature : " + str(result) + " K "
            return render_template("form.html", results=result)
    except:
        return render_template("form.html", results="Invalid Values Entered")


if __name__ == '__main__':
    app.run(debug=True)
