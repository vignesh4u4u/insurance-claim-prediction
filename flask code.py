from flask import Flask,render_template, request
import pickle
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
import warnings
warnings.filterwarnings("ignore")
app = Flask(__name__,template_folder="template")
read=pickle.load(open("insurance_claim.pkl","rb"))
@app.route("/")
def create():
    return render_template("insurance.html")
@app.route("/predict",methods=["POST","GET"])
def predict():
    feature1 = float(request.form["feature1"])
    feature2 = float(request.form["feature2"])
    result = read.predict([[feature1,feature2]])[0]
    return render_template("insurance.html", **locals())

if __name__=="__main__":
    app.run(debug=True)

