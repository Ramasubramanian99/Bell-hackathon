from flask import Flask, request, url_for, redirect, render_template, jsonify
import pickle
import json
import numpy as np

app =Flask(__name__)

loaded_model =  pickle.load(open('./finalized_model.sav', 'rb'))
cols = ['temperature', 'humidity', 'ph', 'rainfall', 'soil_type']

@app.route('/')
def home():
    return 'home'

@app.route('/predict', methods=['POST'])
def predict():
    int_features = [[x for x in request.form.values()]]
    data_unseen = np.array(int_features)
    predicted_value = loaded_model.predict(data_unseen)
    return jsonify(predicted_value[0])


if __name__ == '__main__':
    app.run()