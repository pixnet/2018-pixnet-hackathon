import requests
import base64
import skimage.io as ski_io

from io import BytesIO

def myModel(image):
    return image

# Get Question
endpoint = 'http://pixnethackathon2018-competition.events.pixnet.net/api/question'
payload = {"question_id":1, "img_header":1}

response = requests.get(endpoint, params=payload).json()

image = ski_io.imread(BytesIO(base64.b64decode(response['data']['image'])))
bounding_area = response['data']['bounding_area']


# Do Your Job
answer_image = myModel(image)

# POST Answer
encoded_string = base64.b64encode(answer_image).decode('ascii')

endpoint = 'http://pixnethackathon2018-competition.events.pixnet.net/api/answer'
data = {'question_id': 1, 'key':YOUR_API_KEY, 'image':'data:image/jpeg;base64,' + encoded_string}

response = requests.post(endpoint, json=data).json()
