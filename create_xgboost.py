import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import joblib

# Load datasets
train_data = pd.read_csv("train.csv", encoding="latin-1")
test_data = pd.read_csv("test.csv", encoding="latin-1")

# Preprocess data
train_data['maxUE_UL+DL'] = train_data['maxUE_UL+DL'].replace('#¡VALOR!', np.nan)
train_data['maxUE_UL+DL'] = pd.to_numeric(train_data['maxUE_UL+DL'], errors='coerce')
test_data['maxUE_UL+DL'] = test_data['maxUE_UL+DL'].replace('#¡VALOR!', np.nan)
test_data['maxUE_UL+DL'] = pd.to_numeric(test_data['maxUE_UL+DL'], errors='coerce')

# Encode 'CellName'
le = LabelEncoder()
train_data['CellName_encoded'] = le.fit_transform(train_data['CellName'])
test_data['CellName_encoded'] = le.transform(test_data['CellName'])

# Convert 'Time' to datetime and extract features
train_data['Time'] = pd.to_datetime(train_data['Time'], format='%H:%M')
test_data['Time'] = pd.to_datetime(test_data['Time'], format='%H:%M')
train_data['Hour'] = train_data['Time'].dt.hour
train_data['DayOfWeek'] = train_data['Time'].dt.dayofweek
test_data['Hour'] = test_data['Time'].dt.hour
test_data['DayOfWeek'] = test_data['Time'].dt.dayofweek

# Define features and target
numerical_columns = ['PRBUsageUL', 'PRBUsageDL', 'meanThr_DL', 'meanThr_UL', 'maxThr_DL', 'maxThr_UL', 
                      'meanUE_DL', 'meanUE_UL', 'maxUE_DL', 'maxUE_UL', 'maxUE_UL+DL']
features = numerical_columns + ['CellName_encoded', 'Hour', 'DayOfWeek']

# Check if features are present in both datasets
missing_features_train = [col for col in features if col not in train_data.columns]
missing_features_test = [col for col in features if col not in test_data.columns]

if missing_features_train:
    raise ValueError(f"Missing features in train_data: {missing_features_train}")
if missing_features_test:
    raise ValueError(f"Missing features in test_data: {missing_features_test}")

X_train = train_data[features]
y_train = train_data['Unusual']  # Make sure 'Unusual' exists and is correctly defined in train_data

X_test = test_data[features]

# Define and train XGBoost model
xgb_model = xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
param_grid_xgb = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5],
    'learning_rate': [0.01, 0.1],
    'subsample': [0.8, 1.0]
}
grid_search_xgb = GridSearchCV(xgb_model, param_grid_xgb, cv=5, scoring='f1', n_jobs=-1)
grid_search_xgb.fit(X_train, y_train)
best_xgb_model = grid_search_xgb.best_estimator_

# Save the model
joblib.dump(best_xgb_model, 'flask/models/xgboost_model.h5')

# Print model performance
train_predictions = best_xgb_model.predict(X_train)
accuracy_train = accuracy_score(y_train, train_predictions)
print(f"Training Accuracy: {accuracy_train:.2f}")
