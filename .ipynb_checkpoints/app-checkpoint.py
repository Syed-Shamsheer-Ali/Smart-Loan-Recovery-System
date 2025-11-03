from flask import Flask, render_template, request
import pickle
import numpy as np
import os

# Create Flask app
app = Flask(__name__)

# Load model
model = pickle.load(open("model/loan_model.pkl", "rb"))
scaler = pickle.load(open("model/scaler.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        income = float(request.form['income'])
        loan = float(request.form['loan'])
        tenure = float(request.form['tenure'])
        age = float(request.form['age'])
        input_data = scaler.transform([[income, loan, tenure, age]])
        prediction = model.predict(input_data)[0]
        result = "High Risk" if prediction == 1 else "Low Risk"
        return render_template('result.html', result=result)
    except Exception as e:
        return f"Error: {e}"

if __name__ == '__main__':
    app.run(debug=True)
