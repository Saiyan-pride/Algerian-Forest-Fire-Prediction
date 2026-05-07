from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load model and scaler
model = pickle.load(open('model/model.pkl', 'rb'))
scaler = pickle.load(open('model/scaler.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get values from form
    input_features = [
        float(request.form['Temperature']),
        float(request.form['RH']),
        float(request.form['Ws']),
        float(request.form['Rain']),
        float(request.form['FFMC']),
        float(request.form['DMC']),
        float(request.form['ISI']),
        float(request.form['Classes']),
        float(request.form['Region']),
    ]

    # Scale and predict
    input_array = np.array(input_features).reshape(1, -1)
    scaled_input = scaler.transform(input_array)
    prediction = model.predict(scaled_input)[0]

    return render_template('result.html', prediction=round(prediction, 3))

if __name__ == '__main__':
    app.run(debug=True)