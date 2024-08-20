import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib

# Load the dataset
train_data = pd.read_csv('train.csv', encoding="latin-1")

# Ensure 'CellName' column exists in the dataset
if 'CellName' in train_data.columns:
    # Initialize the Label Encoder
    le = LabelEncoder()
    
    # Fit the encoder on the 'CellName' column
    train_data['CellName_encoded'] = le.fit_transform(train_data['CellName'])
    
    # Save the Label Encoder to a file
    joblib.dump(le, 'flask/models/label_encoder.pkl')
    
    print("Label Encoder has been created and saved as 'label_encoder.pkl'.")
else:
    raise ValueError("The 'CellName' column is missing from the dataset.")
