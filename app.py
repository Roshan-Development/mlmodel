from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Allow all origins or specify React's URL explicitly

# Load the trained model
model = joblib.load("firesafety_model.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        square_feet = data.get('SquareFeet', 1000)
        occupants = data.get('Occupants', 50)
        fire_devices = data.get('NoOfFireSafetyDevicesMinimum', 2)
        exit_doors = data.get('ExitDoors', 2)

        prediction = model.predict([[square_feet, occupants, fire_devices, exit_doors]])[0]
        return jsonify({'SafeOrUnsafe': int(prediction)})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
