import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import os

# -----------------------------
# 1️⃣ Load your dataset
# -----------------------------
df = pd.read_csv("loan-recovery.csv")

# Ensure target column exists
if 'Default' not in df.columns:
    if 'Loan_Status' in df.columns:
        df.rename(columns={'Loan_Status': 'Default'}, inplace=True)
    else:
        raise ValueError("No target column found. Please ensure dataset includes a column like 'Default' or 'Loan_Status'.")


# Fill missing values safely
df = df.fillna(df.median(numeric_only=True))

# -----------------------------
# 2️⃣ Feature Engineering
# -----------------------------
# Example: Create a new "DebtToIncomeRatio"
if "Monthly_Income" in df.columns and "Loan_Amount" in df.columns:
    df["DebtToIncomeRatio"] = df["Loan_Amount"] / (df["Monthly_Income"] + 1)

# Select features (remove Credit_Score if missing)
feature_cols = [col for col in ["Monthly_Income", "Loan_Amount", "Tenure", "Age", "DebtToIncomeRatio"] if col in df.columns]
X = df[feature_cols]
y = df["Default"]

# -----------------------------
# 3️⃣ Split data
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------------
# 4️⃣ Scale features
# -----------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------
# 5️⃣ Random Forest Optimization
# -----------------------------
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf = RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=-1)
grid = GridSearchCV(rf, param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=1)
grid.fit(X_train_scaled, y_train)

best_model = grid.best_estimator_
print("\nBest Parameters:", grid.best_params_)

# -----------------------------
# 6️⃣ Evaluate
# -----------------------------
y_pred = best_model.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {acc * 100:.2f}%")
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# -----------------------------
# 7️⃣ Save model + scaler
# -----------------------------
os.makedirs("model", exist_ok=True)
pickle.dump(best_model, open("model/loan_model.pkl", "wb"))
pickle.dump(scaler, open("model/scaler.pkl", "wb"))

print("\n✅ Model and Scaler saved successfully!")
