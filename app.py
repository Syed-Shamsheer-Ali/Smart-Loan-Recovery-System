from flask import Flask, render_template, request
import numpy as np
import joblib, os

app = Flask(__name__)

# Load trained model and scaler
model = joblib.load(os.path.join("model", "loan_model.pkl"))
scaler = joblib.load(os.path.join("model", "scaler.pkl"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get user input
        income = float(request.form['income'])
        loan = float(request.form['loan'])
        tenure = float(request.form['tenure'])
        age = float(request.form['age'])

        # Prepare input
        data = np.array([[income, loan, tenure, age]])
        data_scaled = scaler.transform(data)

        # Predict
        prediction = model.predict(data_scaled)[0]
        result = "⚠️ High Risk of Default" if prediction == 1 else "✅ Low Risk (Safe)"

        return render_template('result.html', result=result)
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True)
