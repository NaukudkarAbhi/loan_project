import pickle
from flask import Flask,render_template,jsonify,url_for,request,app
import pandas as pd
import numpy as np

app=Flask(__name__)
logmodel=pickle.load(open("final_model","rb"))


@app.route("/")
def home():
    return render_template("home.html")

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data=request.json["data"]
    print(data)
    new_data=np.array(list(data.values())).reshape(1,-1)
    output=logmodel.predict(new_data)
    print(output[0])
    return jsonify(output[0])

@app.route("/predict",methods=["POST"])
def predict():
    data=[float(x) for x in request.form.values()]
    new_data=np.array(list(data.values())).reshape(1,-1)
    print(new_data)
    output=logmodel.predict(new_data)[0]
    return render_template("home.html",prediction_text="status of loan is {}".format(output))

if __name__=="__main__":
    app.run(host='0.0.0.0', port=8080, debug=False)