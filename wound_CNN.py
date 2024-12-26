import pandas as pd
import os
import cv2
import numpy as np
from glob import glob
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt

metadata = pd.read_csv('wound-classification-using-images-and-locations/dataset/Train/wound_locations_Labels_AZH_Train.csv')
image_dir = glob('wound-classification-using-images-and-locations/dataset/Train/images/*')
image_size = (128, 128)

def load_images(image_dirs, image_size):
    images = []
    for dir_path in image_dirs:
        for file in os.listdir(dir_path):
            img_path = os.path.join(dir_path, file)
            if os.path.isfile(img_path):
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.resize(img, image_size)
                    images.append(img)
    return np.array(images)

images = load_images(image_dir, image_size)
images = images / 255.0

label_encoder = LabelEncoder()
location_encoder = LabelEncoder()
labels = label_encoder.fit_transform(metadata['Labels'])  # Assuming 'Labels' is a column for wound labels
locations = location_encoder.fit_transform(metadata['Locations'])  # Assuming 'Locations' is a column for wound locations

metadata_features = metadata.drop(columns=['Labels', 'Locations']).values

X_train_images, X_test_images, X_train_metadata, X_test_metadata, y_train_labels, y_test_labels, y_train_locations, y_test_locations = train_test_split(
    images, metadata_features, labels, locations, test_size=0.2, random_state=42
)

print(f"Training data shapes - Images: {X_train_images.shape}, Metadata: {X_train_metadata.shape}, Labels: {y_train_labels.shape}, Locations: {y_train_locations.shape}")
print(f"Testing data shapes - Images: {X_test_images.shape}, Metadata: {X_test_metadata.shape}, Labels: {y_test_labels.shape}, Locations: {y_test_locations.shape}")

image_input = layers.Input(shape=(128, 128, 3))
x = layers.Conv2D(32, (3, 3), activation='relu')(image_input)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Conv2D(64, (3, 3), activation='relu')(x)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Flatten()(x)
x = layers.Dense(128, activation='relu')(x)
image_output = layers.Dropout(0.5)(x)

metadata_input = layers.Input(shape=(metadata.shape[1] - 2,))
y = layers.Dense(64, activation='relu')(metadata_input)
metadata_output = layers.Dense(32, activation='relu')(y)

combined = layers.concatenate([image_output, metadata_output])

label_output = layers.Dense(len(label_encoder.classes_), activation='softmax', name='label_output')(combined)

location_output = layers.Dense(len(location_encoder.classes_), activation='softmax', name='location_output')(combined)

model = Model(inputs=[image_input, metadata_input], outputs=[label_output, location_output])
model.compile(
    optimizer='adam',
    loss={'label_output': 'sparse_categorical_crossentropy', 'location_output': 'sparse_categorical_crossentropy'},
    metrics={'label_output': 'accuracy', 'location_output': 'accuracy'}
)

model.summary()

X_train_metadata = np.zeros((X_train_images.shape[0], 1), dtype='float32')
X_test_metadata = np.zeros((X_test_images.shape[0], 1), dtype='float32')
print(np.unique(X_train_metadata))
print(f"X_train_images dtype: {X_train_images.dtype}, shape: {X_train_images.shape}")
print(f"X_train_metadata dtype: {X_train_metadata.dtype}, shape: {X_train_metadata.shape}")
print(f"y_train_labels dtype: {y_train_labels.dtype}, shape: {y_train_labels.shape}")
print(f"y_train_locations dtype: {y_train_locations.dtype}, shape: {y_train_locations.shape}")

history = model.fit(
    [X_train_images, X_train_metadata],
    {'label_output': y_train_labels, 'location_output': y_train_locations},
    validation_data=([X_test_images, X_test_metadata], {'label_output': y_test_labels, 'location_output': y_test_locations}),
    epochs=100,
    batch_size=32
)

model.save('wound_classification_model_multi_output.h5')
accuracy_label = history.history['label_output_accuracy']
val_accuracy_label = history.history['val_label_output_accuracy']
loss_label = history.history['label_output_loss']
val_loss_label = history.history['val_label_output_loss']
accuracy_location = history.history['location_output_accuracy']
val_accuracy_location = history.history['val_location_output_accuracy']
loss_location = history.history['location_output_loss']
val_loss_location = history.history['val_location_output_loss']

epochs = range(1, len(accuracy_label) + 1)
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(epochs, accuracy_label, label='Training Accuracy (Labels)')
plt.plot(epochs, val_accuracy_label, label='Validation Accuracy (Labels)')
plt.title('Label Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs, accuracy_location, label='Training Accuracy (Locations)')
plt.plot(epochs, val_accuracy_location, label='Validation Accuracy (Locations)')
plt.title('Location Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(epochs, loss_label, label='Training Loss (Labels)')
plt.plot(epochs, val_loss_label, label='Validation Loss (Labels)')
plt.title('Label Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(epochs, loss_location, label='Training Loss (Locations)')
plt.plot(epochs, val_loss_location, label='Validation Loss (Locations)')
plt.title('Location Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.show()


test_image = cv2.imread('wound-classification-using-images-and-locations/dataset/Test/D/6_0.jpg')
test_image = cv2.resize(test_image, image_size)
test_image = np.expand_dims(test_image / 255.0, axis=0)

test_metadata = pd.read_csv('wound-classification-using-images-and-locations/dataset/Test/wound_locations_Labels_AZH_Test.csv')
print("Test metadata columns:", test_metadata.columns)
single_metadata = np.zeros((1, 1), dtype='float32')
print(f"single_metadata shape: {single_metadata.shape}")

single_prediction = model.predict([test_image, single_metadata])
predicted_label = label_encoder.inverse_transform([np.argmax(single_prediction[0])])
predicted_location = location_encoder.inverse_transform([np.argmax(single_prediction[1])])

print(f"Predicted Label: {predicted_label[0]}, Predicted Location: {predicted_location[0]}")
