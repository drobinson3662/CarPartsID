import customtkinter as ctk
from tkinter import filedialog
import pandas as pd
import numpy as np
import matplotlib
from PIL import Image, ImageTk
from keras._tf_keras.keras.preprocessing import image
from keras._tf_keras.keras.models import load_model
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from data_loader import load_data

#Load the model and labels
model = load_model('car_parts_model.h5')
labels = pd.read_csv('car parts.csv')['labels'].unique()
matplotlib.use('TkAgg')

# Load and preprocess the image
def load_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor /= 255.
    return img_tensor


def test_image():
    global img_path
    img_tensor = load_image(img_path)
    pred = model.predict(img_tensor)
    class_idx = np.argmax(pred, axis=1)[0]
    part_name = labels[class_idx]
    result_label.configure(text=f"This part is likely a: {part_name}")


def upload_image():
    global img_path
    img_path = filedialog.askopenfilename()
    img = Image.open(img_path)
    img.thumbnail((224, 224))
    img = ImageTk.PhotoImage(img)
    image_label.configure(image=img, text="")
    image_label.image = img


def show_confusion_matrix():
    _, _, test_gen = load_data('dataset', img_size=(224, 224), batch_size=32)
    y_true = []
    y_pred = []

    batch_limit = 10
    batch_count = 0
    for images, true_labels in test_gen:
        y_true.extend(np.argmax(true_labels, axis=1))
        predictions = model.predict(images)
        y_pred.extend(np.argmax(predictions, axis=1))

        batch_count += 1
        if batch_count >= batch_limit:
            break

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(12, 12))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap=plt.cm.Blues, ax=ax)
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(fontsize=8)
    plt.title("Confusion Matrix", fontsize=14)
    plt.show()


def show_class_distribution():
    # Load the class labels and count the occurrences
    df = pd.read_csv('car parts.csv')
    class_distribution = df['labels'].value_counts()

    # Plot the distribution
    plt.figure(figsize=(10, 6))
    class_distribution.plot(kind='bar')
    plt.title('Class Distribution')
    plt.xlabel('Class')
    plt.ylabel('Number of Images')
    plt.xticks(rotation=90)
    plt.show()

def show_sample_predictions():
    _, _, test_gen = load_data('dataset', img_size=(224, 224), batch_size=32)
    images, true_labels = next(test_gen)

    predictions = model.predict(images)
    pred_labels = np.argmax(predictions, axis=1)
    true_labels = np.argmax(true_labels, axis=1)

    plt.figure(figsize=(10, 10))

    for i in range(9):  # Display first 9 images
        plt.subplot(3, 3, i + 1)
        plt.imshow(images[i])
        plt.title(f'True: {labels[true_labels[i]]}\nPred: {labels[pred_labels[i]]}')
        plt.axis('off')

    plt.show()

app = ctk.CTk()
app.title("Car Part Identification Program")
app.geometry("800x600")

# Text at top
text_label = ctk.CTkLabel(app, text="Car Part Identification Program", font=("Arial", 20))
text_label.pack(pady=10)

text_label = ctk.CTkLabel(app, text="Please upload an image and click Test to see what you have!")
text_label.pack(pady=10)

# Display the image
image_label = ctk.CTkLabel(app, text="")
image_label.pack(pady=20)

button_frame = ctk.CTkFrame(app, fg_color="transparent")
button_frame.pack(pady=10, fill="x", expand=True)

upload_button = ctk.CTkButton(button_frame, text="Upload Image", command=upload_image)
upload_button.pack(side="left", pady=20, expand=True)

test_button = ctk.CTkButton(button_frame, text="Test Image", command=test_image)
test_button.pack(side="right", pady=20, expand=True)

result_label = ctk.CTkLabel(app, text="")
result_label.pack(pady=20)

visualization_frame = ctk.CTkFrame(app, fg_color="transparent")
visualization_frame.pack(pady=10)

text_label = ctk.CTkLabel(visualization_frame, text="Some commands may take time depending on the speed of your system")
text_label.pack(pady=10)

button1 = ctk.CTkButton(visualization_frame, text="Show Confusion Matrix", command=show_confusion_matrix)
button1.pack(side="left", pady=5, padx=5)

button2 = ctk.CTkButton(visualization_frame, text="Show Class Distribution", command=show_class_distribution)
button2.pack(side="left", pady=5, padx=5)

button3 = ctk.CTkButton(visualization_frame, text="Show Sample Predictions", command=show_sample_predictions)
button3.pack(side="right", pady=5, padx=5)

text_label = ctk.CTkLabel(app, text="Created by Daniel Robinson - 2024")
text_label.pack(side="bottom",  pady=10)

app.mainloop()
