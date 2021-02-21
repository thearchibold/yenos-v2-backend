import re

from flask import Flask, request, render_template
from GenderDetector import detectGender
from flask import jsonify
from flask_cors import CORS


app = Flask(__name__)
CORS(app)


def special_match(strg, search=re.compile(r"[^a-zA-Z\'\"]").search):
    return bool(search(strg))


@app.route('/')
def hello_world():
    return render_template('landing.html')

@app.route('/api_v1/gender/get_gender', methods=['GET'])
@cross_origin()
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


if __name__ == '__main__':
    app.run(debug=True)
