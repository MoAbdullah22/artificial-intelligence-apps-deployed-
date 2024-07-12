from flask import Flask, request, render_template
import numpy as np
import pickle

app = Flask(__name__)

# Load the model and scaler
model_path = 'diabetes-prediction-rfc-model.pkl'
scaler_path = 'scaler.pkl'
with open(model_path, 'rb') as model_file:
    classifier = pickle.load(model_file)
with open(scaler_path, 'rb') as scaler_file:
    sc = pickle.load(scaler_file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    scaled_features = sc.transform(final_features)
    prediction = classifier.predict(scaled_features)
    output = 'You have diabetes' if prediction[0]>0.5 else 'You do not have diabetes'
    return render_template('index.html', prediction_text=output)

if __name__ == '__main__':
    app.run(debug=True)
