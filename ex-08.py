print("727723EUIT223 - SHOBAN CHIDDARTH")

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from flask import Flask, request, jsonify

data = load_diabetes()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_test, check_additivity=False)

shap.summary_plot(shap_values, X_test)
shap.plots.waterfall(shap_values[0])

shap.initjs()
force_plot = shap.plots.force(shap_values[0])

shap.save_html("shap_force_plot.html", force_plot)
print("SHAP interactive visualization saved as shap_force_plot.html")

app = Flask(__name__)

@app.route('/')
def home():
    return "ML Model API is running. Use /predict endpoint."

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = np.array(data["features"]).reshape(1, -1)
    prediction = model.predict(features)
    return jsonify({
        "prediction": float(prediction[0])
    })

if __name__ == '__main__':
    app.run(debug=True)
