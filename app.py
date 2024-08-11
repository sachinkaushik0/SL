from flask import Flask, jsonify, render_template, request
import joblib
import pandas as pd

# Load the pre-trained model and pipeline
best_model = joblib.load('best_model.pkl')
full_pipeline_transformer = joblib.load('full_pipeline_transformer.pkl')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from POST request
        data = request.json

        # Convert data to DataFrame
        query = pd.DataFrame([data])

        # Transform the input data using the pre-loaded pipeline
        query_transformed = full_pipeline_transformer.transform(query)

        # Make prediction using the pre-loaded model
        prediction = best_model.predict(query_transformed)
        probability = best_model.predict_proba(query_transformed)[:, 1]

        # Prepare response
        severity = "Fatal" if prediction[0] == 1 else "Non-Fatal"
        response = {
            "predicted_severity": severity,
            "probability_of_fatal": float(probability[0])
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
