import cv2

from tools.glm import get_treatment_advice
from tools.id_map import get_body_region_by_id,get_wound_type_by_id
from wound_recognizer import analyze_wound


test_image = cv2.imread('wound-classification-using-images-and-locations/dataset/Test/D/48_0.jpg')
wound_id, region_id=analyze_wound(test_image)

body_region=get_body_region_by_id(region_id)
print(body_region)

injury_type=get_wound_type_by_id(wound_id)
print(injury_type)

# injury_type='fracture'
# body_region='arm'
severity='very bad'
user_age=18
user_profession='driver'
location='New York Street 18th'
answer=get_treatment_advice(injury_type,body_region,severity,user_age, user_profession,location)
print(answer)