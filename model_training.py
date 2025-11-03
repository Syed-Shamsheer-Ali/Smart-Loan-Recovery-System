import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Load dataset
df = pd.read_csv("data/loan-recovery.csv")

# ✅ Add Default column if not present
if 'Default' not in df.columns:
    df['Default'] = (df['Num_Missed_Payments'] > 2).astype(int)

# Clean and prepare
df = df.dropna()
df = df[df['Monthly_Income'] > 0]

# Select features that exist in your dataset
X = df[['Monthly_Income', 'Loan_Amount', 'Loan_Tenure', 'Age']]
y = df['Default']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Random Forest
model = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluate
y_pred = model.predict(X_test_scaled)
print("✅ Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save model and scaler
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/loan_model.pkl")
joblib.dump(scaler, "model/scaler.pkl")

print("\n✅ Model and Scaler saved successfully!")
