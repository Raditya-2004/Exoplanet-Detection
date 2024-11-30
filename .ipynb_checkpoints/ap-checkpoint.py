from flask import Flask, request, jsonify
import joblib
import numpy as np

# Load the trained model
model = joblib.load('random_forest_model.pkl')

# Load the label encoder
label_encoder = joblib.load('label_encoder.pkl')  # Save and load the encoder if you used one

app = Flask(__name__)

@app.route('/')
def home():
    return "Exoplanet Detection API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from the request
        data = request.json
        features = [
            float(data['feature1']),
            float(data['feature2']),
            float(data['feature3']),
            float(data['feature4']),
            float(data['feature5']),
            float(data['feature6']),
        ]
        
        # Reshape features for prediction
        features = np.array(features).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(features)
        predicted_class = label_encoder.inverse_transform(prediction)[0]
        
        # Return the result
        return jsonify({'prediction': predicted_class})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
