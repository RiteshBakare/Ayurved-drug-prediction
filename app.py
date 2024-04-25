from flask import Flask, request, jsonify
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier

app = Flask(__name__)

# Load the dataset
df = pd.read_csv('dataset.csv')

# Data Preprocessing
X = df[['disease']]
y = df['drug']

# Encode categorical variable 'disease' using one-hot encoding
encoder = OneHotEncoder()
X_encoded = encoder.fit_transform(X)

# K-Nearest Neighbors Classifier with hyperparameter tuning
knn_clf = KNeighborsClassifier(n_neighbors=5, weights='uniform')  # Initial parameters

# Train the classifier
knn_clf.fit(X_encoded, y)

@app.route('/',methods=['GET'])
def hello():
    return jsonify({'message': 'Hello World!'}), 200

# Route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    input_example = data['input_example']
    input_encoded = encoder.transform([[input_example]])
    predicted_drug = knn_clf.predict(input_encoded)
    return jsonify({'predicted_drug': predicted_drug[0]}), 200

