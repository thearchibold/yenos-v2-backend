import re

from flask import Flask, request, render_template
from GenderDetector import detectGender
from flask import jsonify
from flask_cors import CORS
from Yenos_v2 import Model, Preprocessor


app = Flask(__name__)
CORS(app)

model = Model.Model()
preprocessor = Preprocessor.Preprocessor()


def special_match(strg, search=re.compile(r"[^a-zA-Z\'\"]").search):
    return bool(search(strg))




@app.route('/')
def hello_world():
    return render_template('landing.html')

@app.route('/api_v1/gender/get_gender', methods=['GET'])
def getGender():
    name_data = request.args.get("name")
    if len(name_data) <= 0:
        return jsonify({
            "error":True,
            "message":"Name cannot be empty"
        })
    if special_match(str(name_data)):
         return jsonify({
            "error":True,
            "message":"Invalid Parameter"
        })
    
    else:
       gender = detectGender.predict_gender(name=str(name_data))
       return jsonify(gender)
    pass

@app.route("api/v2/gender", methods=["GET"])
def predict_gender():
    name_data = request.args.get("name")
    if len(name_data) <= 0:
        return jsonify({
            "error":True,
            "message":"Name cannot be empty"
        })
    if special_match(str(name_data)):
         return jsonify({
            "error":True,
            "message":"Invalid Parameter"
        })

    process_data = preprocessor.process_name(name=name_data)
    if process_data is None:
        return jsonify(
            {
            "error":True,
            "message":f"Error processing {name_data}"
        })
        pass
    predictions = model.predict(process_data)

    if predictions is None:
        return jsonify({
            "error":True,
            "message":"Model failed to resolve name"
        })
        pass
    return jsonify({
        "error":False,
        "predictions": predictions
    })







if __name__ == '__main__':
    app.run(debug=True)
