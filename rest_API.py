from flask import Flask, request, redirect, url_for, flash, jsonify
import numpy as np
import pickle as p
import json

from numpy.lib.function_base import vectorize


app = Flask("Job Title Rest_API")


@app.route('/job_title_api/', methods=['POST'])
def makecalc():
    data = request.get_json()
    job_title_vect = vect.transform(data['job_title'])
    prediction = {'Industry':model.predict(job_title_vect)[0]}
    return jsonify(prediction)

if __name__ == '__main__':
    modelfile = './resources/Model/final_prediction.pickle'
    vectorizerfile = './resources/Vectorizer/final_vectorizer.pickle'
    vect = p.load(open(vectorizerfile, 'rb'))
    model = p.load(open(modelfile, 'rb'))
    app.run()