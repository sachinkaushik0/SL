from flask import Flask, jsonify, render_template, request
import joblib
import pandas as pd
import numpy as np

# Load the pre-trained model and pipeline
best_model = joblib.load('/Users/sachinkaushik/accident_severity_predictor/best_model3.pkl')
full_pipeline_transformer = joblib.load('full_pipeline_transformer3.pkl')

app = Flask(__name__)   

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from POST request
        data = request.json
        print(f"Received data: {data}")

        # Convert data to DataFrame
        query = pd.DataFrame([data])
        print(f"DataFrame query: {query}")

        # Transform the input data using the pre-loaded pipeline
        query_transformed = full_pipeline_transformer.transform(query)
        print(f"Transformed query shape: {query_transformed.shape}")

        # Make prediction using the pre-loaded model
        prediction = best_model.predict(query_transformed)
        probability = best_model.predict_proba(query_transformed)[:, 1]
        print(f"Raw prediction: {prediction}, Probability: {probability}")

        # Ensure prediction is a single value
        prediction = prediction[0]
        probability = probability[0]

        # Prepare response
        severity = "Fatal" if prediction == 1 else "Non-Fatal"
        response = {
            "predicted_severity": severity,
            "probability_of_fatal": float(probability),
            "raw_prediction": int(prediction)
        }
        
        print(f"Sending response: {response}")

        return jsonify(response)

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)