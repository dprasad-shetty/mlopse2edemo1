from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load the pre-trained logistic regression model
with open('logistic_regression_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/')
def home():
    return "Logistic Regression Model is ready to make predictions."

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract JSON data from the request
        data = request.get_json(force=True)
        
        # Convert data into a numpy array
        input_features = np.array([data['features']])
        
        # Make prediction
        prediction = model.predict(input_features)
        
        # Return the prediction as a JSON response
        return jsonify({'prediction': int(prediction[0])})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
