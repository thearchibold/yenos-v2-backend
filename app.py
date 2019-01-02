import re

from flask import Flask, request, render_template

from GenderDetector import detectGender

app = Flask(__name__)


def special_match(strg, search=re.compile(r"[^a-zA-Z\'\"]").search):
    return bool(search(strg))


@app.route('/')
def hello_world():
    return render_template('landing.html')

@app.route('/api_v1/gender/get_gender', methods=['GET'])
def getGender():
    name_data = request.args.get("name")
    if len(name_data) <= 0:
        return "Name cannot be empty"
    if special_match(str(name_data)):
        return "Invalid Parameter"

    else:
       gender = detectGender.predict_gender(name=str(name_data))
       return str(gender)


if __name__ == '__main__':
    app.run(debug=True)
