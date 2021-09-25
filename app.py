from flask import Flask, request, url_for, redirect, render_template, jsonify
import pickle
import json
import numpy as np

app =Flask(__name__)

loaded_model =  pickle.load(open('./finalized_model.sav', 'rb'))
cols = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall', 'soil_type']

@app.route('/')
def home():
    return 'home.html'

@app.route('/predict', methods=['POST'])
def predict():
    int_features = [[x for x in request.form.values()]]
    int_features =[[90, 42, 43, 20.87974371, 82.00274423, 6.502985292, 202.9355362, 0]]
    data_unseen = np.array(int_features)
    predicted_value = loaded_model.predict(data_unseen)
    return str(predicted_value[0])


if __name__ == '__main__':
    app.run()