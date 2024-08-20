from flask import Flask, request, render_template, send_file, redirect, url_for
import pandas as pd
import xgboost as xgb
import joblib
import os

app = Flask(__name__)

def detect_anomalies(csv_file_path, model_file_path):
    # Load the dataset
    data = pd.read_csv(csv_file_path, encoding="latin-1")
    
    # Preprocess data
    data['maxUE_UL+DL'] = data['maxUE_UL+DL'].replace('#¡VALOR!', pd.NA)
    data['maxUE_UL+DL'] = pd.to_numeric(data['maxUE_UL+DL'], errors='coerce')
    
    # Encode 'CellName'
    if 'CellName' in data.columns:
        le = joblib.load('models/label_encoder.pkl')  # Load label encoder
        data['CellName_encoded'] = le.transform(data['CellName'])
    else:
        raise ValueError("The 'CellName' column is missing from the input data.")

    # Convert 'Time' to datetime and extract features
    if 'Time' in data.columns:
        data['Time'] = pd.to_datetime(data['Time'], format='%H:%M')
        data['Hour'] = data['Time'].dt.hour
        data['DayOfWeek'] = data['Time'].dt.dayofweek
    else:
        raise ValueError("The 'Time' column is missing from the input data.")

    # Define features
    features = ['PRBUsageUL', 'PRBUsageDL', 'meanThr_DL', 'meanThr_UL', 'maxThr_DL', 'maxThr_UL', 
                'meanUE_DL', 'meanUE_UL', 'maxUE_DL', 'maxUE_UL', 'maxUE_UL+DL', 'CellName_encoded', 'Hour', 'DayOfWeek']
    
    # Check if features are present in the dataset
    missing_features = [col for col in features if col not in data.columns]
    if missing_features:
        raise ValueError(f"Missing features in data: {missing_features}")
    
    # Extract features
    X = data[features]

    # Load the model
    model = joblib.load(model_file_path)
    
    # Predict anomalies
    data['Anomaly'] = model.predict(X)
    data['Anomaly_Probability'] = model.predict_proba(X)[:, 1]
    
    # Filter rows with anomalies
    anomalies = data[data['Anomaly'] == 1]
    
    return anomalies, data

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            file_path = os.path.join('uploads', file.filename)
            file.save(file_path)
            
            model_file_path = 'models/xgboost_model.h5'
            anomalies, data = detect_anomalies(file_path, model_file_path)
            
            # Save CSV without anomalies
            no_anomalies_path = 'downloads/no_anomalies.csv'
            data[data['Anomaly'] == 0].to_csv(no_anomalies_path, index=False)
            
            # Store anomalies in a CSV
            anomalies.to_csv('downloads/anomalies.csv', index=False)
            
            return redirect(url_for('result', page=1))
    
    return render_template('index.html')

@app.route('/result', methods=['GET', 'POST'])
def result():
    page = int(request.args.get('page', 1))
    per_page = 100
    min_prob = float(request.args.get('min_prob', 0))
    max_prob = float(request.args.get('max_prob', 1))  # Default to 1 if max_prob is not provided
    
    anomalies_file_path = 'downloads/anomalies.csv'
    anomalies = pd.read_csv(anomalies_file_path)
    
    # Filter anomalies based on probability range
    if min_prob < max_prob:
        anomalies = anomalies[(anomalies['Anomaly_Probability'] >= min_prob) & (anomalies['Anomaly_Probability'] < max_prob)]
    
    # Reorder columns: move 'Anomaly_Probability' to the first column and drop 'Anomaly'
    anomalies = anomalies[['Anomaly_Probability'] + [col for col in anomalies.columns if col != 'Anomaly_Probability' and col != 'Anomaly']]

    total_entries = len(anomalies)
    start = (page - 1) * per_page
    end = start + per_page
    
    paginated_anomalies = anomalies[start:end]
    
    return render_template('result.html', 
                          anomalies=paginated_anomalies.to_html(classes='table table-striped', index=False), 
                          count=len(paginated_anomalies),
                          total_entries=total_entries,
                          current_page=page,
                          total_pages=(total_entries + per_page - 1) // per_page,
                          min_prob=min_prob,
                          max_prob=max_prob)

@app.route('/download')
def download_file():
    return send_file('downloads/no_anomalies.csv', as_attachment=True)

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    if not os.path.exists('downloads'):
        os.makedirs('downloads')
    app.run(debug=True)
