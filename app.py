from flask import Flask, make_response
from flask import request
from tools.glm import get_treatment_advice

app = Flask(__name__)


@app.route('/test_QA', methods=['POST'])
def custom_response():
    injury_type = request.form.get('injury_type')
    body_region = request.form.get('body_region')
    severity = request.form.get('severity')
    user_age = 18
    user_profession = 'driver'
    answer = get_treatment_advice(injury_type, body_region, severity, user_age, user_profession)

    response = make_response('This is an answer from chatglm')
    response.data = answer
    response.status_code = 200
    print(response)
    return response


if __name__ == '__main__':
    app.run(debug=False)
