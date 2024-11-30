from flask import Flask, request, jsonify, send_from_directory, redirect
import joblib
import numpy as np

# Load the trained model
model = joblib.load('random_forset_model.pkl')

app = Flask(__name__)

@app.route('/')
def home():
    return redirect('/form')

@app.route('/form')
def form():
    return send_from_directory('static', 'index.html')

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
            float(data['feature7']),
            float(data['feature8']),
        ]
        
        # Reshape features for prediction
        features = np.array(features).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(features)
        
        # If label_encoder is not used:
        if prediction[0]==0 :
            predicted_class = "FALSE FOUND EXOPLANET"
        elif prediction[0]==1 :
            predicted_class = "CONFIRMED EXOPLANET"
        else:
            predicted_class = "CANDATE EXOPLANET"
        
        # If label_encoder is used:
        # predicted_class = label_encoder.inverse_transform(prediction)[0]
        
        return jsonify({'prediction': str(predicted_class)})
    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True)
