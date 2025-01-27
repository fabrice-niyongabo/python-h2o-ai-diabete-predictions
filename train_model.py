import h2o
from h2o.automl import H2OAutoML
import pandas as pd

# Start H2O server
h2o.init()

# Load the dataset
data = h2o.import_file('diabetes_data.csv')

# Split data into training and testing sets
train, test = data.split_frame(ratios=[0.8])

# Define target and features
y = 'diabetes'  # Target column
x = data.columns.remove(y)  # All other columns are features

# Train the model using AutoML
aml = H2OAutoML(max_models=5, seed=42)
aml.train(x=x, y=y, training_frame=train)

# Save the best model
model_path = h2o.save_model(aml.leader, path="./diabetes_model", force=True)
print(f"Model saved at {model_path}")
