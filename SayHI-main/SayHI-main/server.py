import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS, cross_origin
import pickle
from predict2 import final
import soundfile
import pandas as pd
from sklearn.model_selection import train_test_split

app = Flask(__name__)
CORS(app)
model = pickle.load(open('gender2.pkl','rb'))
@app.route("/", methods=["GET"])
def main():
    return render_template("index.html")

@app.route('/login')
def login():
    return render_template("login.html")

@app.route('/upload')
def upload():
    return render_template("upload.html")

@app.route('/record')
def record():
    return render_template("record.html")

@app.route('/signup')
def signup():
    return render_template("signup.html")

@app.route('/api',methods=['POST'])
def predict():
    try:
        data = request.files['audioFile']
        print(data.filename)
    except Exception as e:
        print(f"{e} happened.")
    
    fn = data.filename.split(".")[0]
#    sf, sr = soundfile.read(data)
    with open(f"{fn}.wav","wb") as f:
        data.save(f)
    feat = final()
    print("feat rcvd")
    # columns = ['tonnetz1_mean', 'tonnetz2_mean', 'tonnetz3_mean', 'tonnetz4_mean', 
    #        'tonnetz5_mean', 'tonnetz6_mean', 'tonnetz1_var', 'tonnetz2_var', 
    #        'tonnetz3_var', 'tonnetz4_var', 'tonnetz5_var', 'tonnetz6_var', 
    #        'spec_centroid_mean', 'spec_centroid_var', 
    #        'mfcc1_mean', 'mfcc2_mean', 'mfcc3_mean', 'mfcc4_mean', 'mfcc5_mean', 
    #        'mfcc6_mean', 'mfcc7_mean', 'mfcc8_mean', 'mfcc9_mean', 'mfcc10_mean', 
    #        'mfcc11_mean', 'mfcc12_mean', 'mfcc13_mean', 'mfcc14_mean', 'mfcc15_mean', 
    #        'mfcc16_mean', 'mfcc17_mean', 'mfcc18_mean', 'mfcc19_mean', 'mfcc20_mean',
    #        'mfcc1_var', 'mfcc2_var', 'mfcc3_var', 'mfcc4_var', 'mfcc5_var', 
    #        'mfcc6_var', 'mfcc7_var', 'mfcc8_var', 'mfcc9_var', 'mfcc10_var', 
    #        'mfcc11_var', 'mfcc12_var', 'mfcc13_var', 'mfcc14_var', 'mfcc15_var', 
    #        'mfcc16_var', 'mfcc17_var', 'mfcc18_var', 'mfcc19_var', 'mfcc20_var',
    #        'spec_width_mean', 'spec_width_var', 'spec_contrast1_mean', 'spec_contrast2_mean', 
    #        'spec_contrast3_mean', 'spec_contrast4_mean', 'spec_contrast5_mean', 'spec_contrast6_mean', 
    #        'spec_contrast7_mean', 'spec_contrast_var1', 'spec_contrast2_var', 'spec_contrast3_var', 
    #        'spec_contrast4_var', 'spec_contrast5_var', 'spec_contrast6_var', 'spec_contrast7_var'
    #       ]
    feat = [feat]
    print(feat)
    print(model.classes_)
    label = model.predict(feat)
    return render_template("upload.html",message=label[0])
if __name__ == '__main__':
    app.run(port=5000, debug=True)