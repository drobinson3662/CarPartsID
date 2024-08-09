import customtkinter as ctk
from tkinter import filedialog
import pandas as pd
import numpy as np
from PIL import Image, ImageTk
from keras._tf_keras.keras.preprocessing import image
from keras._tf_keras.keras.models import load_model

#Load the model and labels
model = load_model('car_parts_model.h5')
labels = pd.read_csv('car parts.csv')['labels'].unique()

# Load and preprocess the image
def load_image(img_path):
    img= image.load_img(img_path, target_size=(224, 224))
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
    img.thumbnail((400, 400))
    img = ImageTk.PhotoImage(img)
    image_label.configure(image=img, text="")
    image_label.image=img

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

text_label = ctk.CTkLabel(app, text="Created by Daniel Robinson - 2024")
text_label.pack(pady=10)

app.mainloop()