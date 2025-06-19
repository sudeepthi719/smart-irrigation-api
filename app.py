from flask import Flask, request, jsonify
from flask_cors import CORS  # <-- Add this import
import joblib

app = Flask(__name__)
CORS(app)
# Load models and encoders
clf_irrigation = joblib.load("clf_irrigation.pkl")
reg_water = joblib.load("reg_water.pkl")
clf_crop_after = joblib.load("clf_crop_after.pkl")
clf_crop_without = joblib.load("clf_crop_without.pkl")
le_irrigation = joblib.load("le_irrigation.pkl")
le_crop_after = joblib.load("le_crop_after.pkl")
le_crop_without = joblib.load("le_crop_without.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    user_input = [[
        data['soil_moisture'],
        data['humidity'],
        data['temperature'],
        data['rainfall']
    ]]
    irrigation_pred = clf_irrigation.predict(user_input)[0]
    irrigation_label = le_irrigation.inverse_transform([irrigation_pred])[0]
    water_pred = float(reg_water.predict(user_input)[0]) if irrigation_label == "Yes" else 0.0
    crop_after_pred = clf_crop_after.predict(user_input)[0]
    crop_without_pred = clf_crop_without.predict(user_input)[0]
    crop_after_label = le_crop_after.inverse_transform([crop_after_pred])[0]
    crop_without_label = le_crop_without.inverse_transform([crop_without_pred])[0]
    return jsonify({
        "irrigation_needed": irrigation_label,
        "water_required": water_pred,
        "best_crop_after_irrigation": crop_after_label,
        "best_crop_without_irrigation": crop_without_label
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
