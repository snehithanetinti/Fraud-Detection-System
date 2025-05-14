import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, f1_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# 1. Create sample financial transaction data
data = pd.DataFrame({
    'amount': [100, 1500, 2500, 300, 4500, 90, 1200, 800, 4000, 60],
    'transaction_type_encoded': [0, 1, 1, 0, 1, 0, 1, 0, 1, 0],
    'is_fraud': [0, 0, 1, 0, 1, 0, 0, 0, 1, 0]  # Contains both 0 and 1
})

print("Data Preview:")
print(data)

# 2. Separate features and labels
features = data.drop(['is_fraud'], axis=1)
labels = data['is_fraud']

# 3. Standardize features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# 4. Handle class imbalance using SMOTE (Adjust n_neighbors to 2)
smote = SMOTE(random_state=42, k_neighbors=2)  # Ensure n_neighbors is <= number of fraud samples
X_resampled, y_resampled = smote.fit_resample(features_scaled, labels)

# 5. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42
)

# 6. Isolation Forest (Anomaly Detection)
isolation_forest = IsolationForest(contamination=0.2, random_state=42)
isolation_forest.fit(X_train)

# Predict: -1 means anomaly -> convert to 1 (fraud)
y_pred_if = isolation_forest.predict(X_test)
y_pred_if = [1 if pred == -1 else 0 for pred in y_pred_if]

print("\nIsolation Forest Results:")
print(classification_report(y_test, y_pred_if))
print("F1 Score:", f1_score(y_test, y_pred_if))
print("AUC-ROC:", roc_auc_score(y_test, y_pred_if))

# 7. Random Forest Classifier (Supervised)
rfc = RandomForestClassifier(n_estimators=100, random_state=42)
rfc.fit(X_train, y_train)
y_pred_rfc = rfc.predict(X_test)

print("\nRandom Forest Classifier Results:")
print(classification_report(y_test, y_pred_rfc))
print("F1 Score:", f1_score(y_test, y_pred_rfc))
print("AUC-ROC:", roc_auc_score(y_test, y_pred_rfc))