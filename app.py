from flask import Flask, render_template, request
import pickle
import numpy as np

# Load the model
model_path = 'model.pkl'


with open(model_path, 'rb') as file:
    model = pickle.load(file)

app = Flask(__name__)
# Define the home route
@app.route('/')
def home():
    return render_template('index.html')

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    # Get input values from the form
    input_features = [float(x) for x in request.form.values()]
    
    # Convert to numpy array and reshape
    input_array = np.array(input_features).reshape(1, -1)
    
    # Make prediction using the loaded model
    prediction = model.predict(input_array)
    
    # Return the result on the web page
    return render_template('index.html', prediction_text=f'Predicted Species: {prediction[0]}')

if __name__ == '__main__':
    app.run(debug=True)
