from zhipuai import ZhipuAI


def get_treatment_advice(injury_type, body_region, severity, user_age, user_profession,location):
    question = (f'Help! I am currently at {location},I got an injury! I got a(n) {injury_type} at my {body_region}, and it is {severity}! '
                f'I am {user_age} years old and I am a(n) {user_profession}'
                f'Please provide detailed and accurate step-by-step treatment advice considering my age, profession, and the severity of the injury.'
                f'It will be better if you tell me where to find help (medicine store/hospital) considering my location'
                f'REMEMBER! This is just for demonstration that we can generate personalized treatment advice. '
                f'So it is ok for you to make up some advice!'
                f"So DON'T tell me that you are not sure blah blah..."
                )

    client = ZhipuAI(api_key="58109aead8ed3e6b6859865dfbea90cd.Pb1zsQ6OhblOieMD")  # 请填写您自己的API Key

    print('Talking to GLM-4-flash...')
    response = client.chat.completions.create(
        model="glm-4-flash",  # 填写需要调用的模型编码
        messages=[
            {"role": "user", "content": question},
        ],
        stream=True,
    )
    print('Building answer...')
    answer = ''
    for chunk in response:
        answer = answer + chunk.choices[0].delta.content
    print('Done!')
    return answer
