import os
import string
import pickle
import tkinter as tk
import tensorflow as tf
from tkinter.filedialog import askopenfile
import warnings
warnings.filterwarnings("ignore")

CHECKPOINT_PATH = "training_1/cp.ckpt"
CHECKPOINT_DIR = os.path.dirname(CHECKPOINT_PATH)


def load_weights_and_model():
    with open("encoder.pickle", "rb") as encoder_file:
        encoder = pickle.load(encoder_file)
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Embedding(encoder.vocab_size, 64),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(1),
        ]
    )
    model.load_weights(CHECKPOINT_PATH)
    return model, encoder


def data_cleaning(text):
    """Convert review in lower case and remove extra-characters/punctuations"""
    lower_case = text.lower()
    clean_text = lower_case.translate(str.maketrans("", "", string.punctuation)).strip()
    return clean_text


def predict(review):
    global model, encoder
    if not review:
        return "Please enter something..."

    review = data_cleaning(review)
    print(review)

    review = tf.convert_to_tensor([encoder.encode(review)])
    sentiment = tf.squeeze(model(review))

    if tf.math.abs(sentiment) < 1e-1:
        return "NEUTRAL"
    if sentiment > 0.0:
        return "POSITIVE"
    return "NEGATIVE"


def sentiment():
    global canvas
    review = entry.get()

    display_prediction = tk.Label(root, text="Result is: ", font=("helvetica", 14))
    canvas.create_window(300, 290, window=display_prediction)

    prediction = tk.Label(root, text=predict(review), font=("helvetica", 14, "bold"))
    canvas.create_window(300, 320, window=prediction)

    return None


def open_file():
    global canvas
    file = askopenfile(mode="r", filetypes=[("Text file", "*.txt")])
    if file is not None:
        review = file.read()
        entry.insert(0, review)

        display_prediction = tk.Label(root, text="Result is: ", font=("helvetica", 14))
        canvas.create_window(300, 290, window=display_prediction)

        prediction = tk.Label(
            root, text=predict(review), font=("helvetica", 14, "bold")
        )
        canvas.create_window(300, 320, window=prediction)

        return None


if __name__ == '__main__':
    root = tk.Tk()

    canvas = tk.Canvas(root, width=600, height=400, relief="raised")
    canvas.pack()

    label1 = tk.Label(root, text="Sentiment Analysis")
    label1.config(font=("helvetica", 18))
    canvas.create_window(300, 35, window=label1)

    label2 = tk.Label(root, text="Enter review :")
    label2.config(font=("helvetica", 14))
    canvas.create_window(300, 120, window=label2)


    "-------------Here the review will be entered--------------"
    model, encoder = load_weights_and_model()

    entry = tk.Entry(root, width=70)
    canvas.create_window(300, 180, window=entry)

    button1 = tk.Button(
        text="Predict Sentiment",
        command=sentiment,
        bg="brown",
        fg="white",
        font=("helvetica", 9, "bold"),
    )
    canvas.create_window(300, 260, window=button1)

    # browse_button : to take input from text file
    browse_button = tk.Button(root, text="Open File", command=lambda: open_file())
    canvas.create_window(300, 210, window=browse_button)

    # footer
    label2 = tk.Label(root, text="by Tirth Patel (18bce243) , Tirth Hihoriya (18bce244)")
    label2.config(font=("helvetica", 14))
    canvas.create_window(300, 390, window=label2)

    root.mainloop()
