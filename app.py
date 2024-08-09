import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
from keras._tf_keras.keras.preprocessing import image
from keras._tf_keras.keras.models import load_model
import customtkinter as ctk
from tkinter import filedialog


def load_image(img_path, show=False):
    img = image.load_img(img_path, target_size=(224, 224))
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor /= 255.

    if show:
        plt.imshow(img_tensor[0])
        plt.axis('off')
        plt.show()

    return img_tensor

def load_labels(csv_path):
    df = pd.read_csv(csv_path)
    return df['labels'].unique()

if __name__ == '__main__':
    model_path = 'car_parts_model.h5'
    model = load_model(model_path)
    # custom_objects = {"EfficientNetB2": EfficientNetB2,}
    # model = load_model(model_path, custom_objects=custom_objects, compile=False)

    csv_path = 'car parts.csv'
    labels = load_labels(csv_path)
    # print("Labels loaded:", labels)

    img_path = sys.argv[1]
    new_image = load_image(img_path, show=True)

    pred = model.predict(new_image)
    class_idx = np.argmax(pred, axis=1)[0]

    # print("Predection:", pred)
    # print("Class Index", class_idx)

    part_name = labels[class_idx]

    print(f'This image is likely a: {part_name}')



