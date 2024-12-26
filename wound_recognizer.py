import tensorflow as tf
import cv2
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def analyze_wound(test_image):
    image_size = (128, 128)
    metadata = pd.read_csv('wound-classification-using-images-and-locations/dataset/Train/wound_locations_Labels_AZH_Train.csv')

    model = tf.keras.models.load_model('model.h5')

    model.summary()

    label_encoder = LabelEncoder()
    location_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(metadata['Labels'])
    locations = location_encoder.fit_transform(metadata['Locations'])

    # test_image = cv2.imread('wound-classification-using-images-and-locations/dataset/Test/D/48_0.jpg')
    test_image = cv2.resize(test_image, image_size)
    test_image = np.expand_dims(test_image / 255.0, axis=0)

    test_metadata = pd.read_csv('wound-classification-using-images-and-locations/dataset/Test/wound_locations_Labels_AZH_Test.csv')
    print("Test metadata columns:", test_metadata.columns)
    single_metadata = np.zeros((1, 1), dtype='float32')
    print(f"single_metadata shape: {single_metadata.shape}")

    single_prediction = model.predict([test_image, single_metadata],batch_size=32)
    predicted_label = label_encoder.inverse_transform([np.argmax(single_prediction[0])])
    predicted_location = location_encoder.inverse_transform([np.argmax(single_prediction[1])])

    print(f"Predicted Label: {predicted_label[0]}, Predicted Location: {predicted_location[0]}")
    return predicted_label[0],predicted_location[0]