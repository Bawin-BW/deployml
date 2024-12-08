from flask import Flask, request, jsonify
import pickle
from tensorflow.keras.models import load_model
import numpy as np

app = Flask(__name__)

# Load the model and scalers
loaded_model = load_model('student_performance__fix_model.h5')
with open('scaler_X.pkl', 'rb') as f:
    scaler_X = pickle.load(f)
with open('scaler_y.pkl', 'rb') as f:
    scaler_y = pickle.load(f)

def predict_performance(model, scaler_X, scaler_y, Hours_Studied, Sleep_Hours, Previous_Scores, Sample_Question_Papers_Practiced, Extracurricular_Activities):
    # Input validation
    if not 0 <= Hours_Studied <= 24:
        return jsonify({'error': 'Hours Studied must be between 0 and 24'}), 400
    if not 0 <= Sleep_Hours <= 24:
        return jsonify({'error': 'Sleep Hours must be between 0 and 24'}), 400
    if not 0 <= Previous_Scores <= 100:
        return jsonify({'error': 'Previous Scores must be between 0 and 100'}), 400
    if not 0 <= Sample_Question_Papers_Practiced <= 100:
        return jsonify({'error': 'Sample Question Papers Practiced must be between 0 and 100'}), 400
    if Extracurricular_Activities not in ['Yes', 'No']:
        return jsonify({'error': 'Extracurricular Activities must be either "Yes" or "No"'}), 400

    # Convert extracurricular activity to numeric
    Extracurricular = 1 if Extracurricular_Activities == 'Yes' else 0
    input_data = [[Hours_Studied, Sleep_Hours, Previous_Scores, Sample_Question_Papers_Practiced, Extracurricular]]
    input_data_scaled = scaler_X.transform(input_data)
    
    # Get the prediction from the model
    prediction_scaled = model.predict(input_data_scaled)
    
    # Inverse scaling to get the actual prediction value
    prediction = scaler_y.inverse_transform(prediction_scaled)[0][0]
    
    # Ensure the prediction is capped between 0 and 100
    prediction = max(0, min(100, prediction))
    
    return prediction

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse incoming JSON data
        data = request.get_json()

        # Extract features from the incoming data
        hours_studied = float(data['Hours_Studied'])
        sleep_hours = float(data['Sleep_Hours'])
        previous_scores = float(data['Previous_Scores'])
        sample_papers = float(data['Sample_Question_Papers_Practiced'])
        extracurricular = data['Extracurricular_Activities']

        # Call the prediction function
        prediction = predict_performance(loaded_model, scaler_X, scaler_y, hours_studied, sleep_hours, previous_scores, sample_papers, extracurricular)
        
        # Return the prediction as a JSON response
        return jsonify({'prediction': float(prediction)})  # Ensure the prediction is a native float
    
    except KeyError as e:
        return jsonify({'error': f'Missing required field: {str(e)}'}), 400
    except ValueError as e:
        return jsonify({'error': f'Invalid value provided: {str(e)}'}), 400
    except TypeError as e:
        return jsonify({'error': f'Incorrect type provided: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)