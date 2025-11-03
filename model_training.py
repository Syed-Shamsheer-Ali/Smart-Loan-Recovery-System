import pandas as pd
import numpy as np
import os
import warnings
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

warnings.filterwarnings("ignore")


df = pd.read_csv("data/loan-recovery.csv")


df = df.dropna()
df = df[df['Monthly_Income'] > 0]

features = [
    'Age', 'Monthly_Income', 'Num_Dependents', 'Loan_Amount',
    'Loan_Tenure', 'Interest_Rate', 'Collateral_Value',
    'Outstanding_Loan_Amount', 'Monthly_EMI',
    'Num_Missed_Payments', 'Days_Past_Due', 'Collection_Attempts'
]

if 'Default' not in df.columns:
    df['Default'] = np.where(
        (df['Num_Missed_Payments'] > 3) |
        (df['Days_Past_Due'] > 90) |
        (df['Collection_Attempts'] > 4),
        1, 0
    )

if 'Default' not in df.columns:
    raise ValueError("Dataset must include a 'Default' column (0 = repaid, 1 = defaulted).")


X = df[features]
y = df['Default']


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


rf = RandomForestClassifier(random_state=42)

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [6, 8, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}


grid = GridSearchCV(
    rf, param_grid, cv=3, scoring='accuracy', verbose=1, n_jobs=-1
)
grid.fit(X_train_scaled, y_train)

best_rf = grid.best_estimator_
print(f"\nBest Parameters: {grid.best_params_}")


y_pred = best_rf.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)

print(f"\nModel Accuracy: {round(acc * 100, 2)}%")
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


os.makedirs("model", exist_ok=True)
joblib.dump(best_rf, "model/loan_model.pkl")
joblib.dump(scaler, "model/scaler.pkl")

print("\nRandom Forest model and scaler saved successfully!")
