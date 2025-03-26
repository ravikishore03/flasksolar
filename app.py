#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load the trained model and scaler
model = joblib.load("solar_model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get input values from the form
        features = [
            float(request.form["distance-to-solar-noon"]),
            float(request.form["temperature"]),
            float(request.form["wind-direction"]),
            float(request.form["wind-speed"]),
            float(request.form["sky-cover"]),
            float(request.form["visibility"]),
            float(request.form["humidity"]),
            float(request.form["average-wind-speed"]),
            float(request.form["average-pressure"])
        ]

        # Prepare the input for the model
        input_data = np.array([features])
        input_scaled = scaler.transform(input_data)

        # Make a prediction and map to category
        prediction = model.predict(input_scaled)[0]
        categories = ["Low", "Medium", "High"]
        
        return render_template("result.html", prediction=categories[prediction])
    
    except Exception as e:
        return render_template("error.html", error=str(e))

if __name__ == "__main__":
    app.run(debug=True)

