import re

from flask import Flask, request

from GenderDetector import detectGender

app = Flask(__name__)


def special_match(strg, search=re.compile(r"[^a-zA-Z]").search):
    return bool(search(strg))


@app.route('/')
def hello_world():
    return 'Welcome to the Gender Generator API. This system predicts a persons gender based on christian or first name. Made with love by Archibold Bernard!'

@app.route('/api_v1/gender/get_gender', methods=['GET'])
def getGender():
    name_data = request.args.get("name")

    if special_match(str(name_data)):
        return "Invalid Parameter"

    else:
       gender = detectGender.predict_gender(name=str(name_data))
       return str(gender)


if __name__ == '__main__':
    app.run()
