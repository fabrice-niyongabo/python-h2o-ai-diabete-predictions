from flask import Flask, request, jsonify
import h2o
import pandas as pd
from h2o.frame import H2OFrame

# Initialize Flask app
app = Flask(__name__)

# Start H2O server
h2o.init()

# Load the saved model
model_path = './diabetes_model/GLM_1_AutoML_1_20250127_151855'  # Replace with the actual file name
model = h2o.load_model(model_path)

@app.route('/predict', methods=['POST'])
def predict():
    # Parse incoming JSON data
    data = request.json
    # Example input: {"carbs": 60, "fats": 15, "proteins": 25, "sugars": 30}
    input_data = pd.DataFrame([data])
    
    # Convert to H2OFrame
    h2o_data = H2OFrame(input_data)
    
    # Make predictions
    predictions = model.predict(h2o_data)
    
    # Extract prediction result
    prediction_result = int(predictions.as_data_frame()['predict'][0])
    return jsonify({'diabetes_risk': prediction_result})

if __name__ == '__main__':
    app.run(debug=True)
