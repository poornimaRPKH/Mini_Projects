#Part 1 — Import required libraries:
from flask import Flask, request, jsonify
import pickle
import numpy as np

#Part 2 — Load the saved model and create the Flask app:
# Load the trained model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# Create Flask app
app = Flask(__name__)
#Part 3 — Create the prediction endpoint:
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # Extract features from the incoming data
    features = np.array([[
        data['Pclass'],
        data['Sex'],
        data['Age'],
        data['SibSp'],
        data['Parch'],
        data['Fare'],
        data['Embarked_Q'],
        data['Embarked_S']
    ]])

    # Make prediction
    prediction = model.predict(features)

    return jsonify({'Survived': int(prediction[0])})
#Part 4 — Run the Flask app:
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True) 