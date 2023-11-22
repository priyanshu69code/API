from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from typing import List
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.sequence import pad_sequences
import pickle
from keras.applications.inception_v3 import InceptionV3
from keras.models import Model

app = FastAPI()

# Load the image captioning model
captioning_model = load_model("main_model.h5")

# Load the InceptionV3 model for image encoding
inception_model = InceptionV3(weights='imagenet')

# Create a new model, by removing the last layer (output layer) from the inception v3
inception_model_new = Model(inception_model.input,
                            inception_model.layers[-2].output)

# Load the dictionaries
with open("wordtoix.pkl", "rb") as encoded_pickle:
    wordtoix = pickle.load(encoded_pickle)

with open("ixtoword.pkl", "rb") as encoded_pickle:
    ixtoword = pickle.load(encoded_pickle)

max_length = 33


def preprocess(image_path):
    # Convert all the images to size 299x299 as expected by the inception v3 model
    img = image.load_img(image_path, target_size=(299, 299))
    # Convert PIL image to numpy array of 3-dimensions
    x = image.img_to_array(img)
    # Add one more dimension
    x = np.expand_dims(x, axis=0)
    # preprocess the images using preprocess_input() from inception module
    x = preprocess_input(x)
    return x


def encode_image(image_path):
    # Load and preprocess the image
    img = image.load_img(image_path, target_size=(299, 299))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Get the image encoding using the pre-trained InceptionV3 model
    img_encoding = inception_model_new.predict(img_array)
    img_encoding = np.reshape(img_encoding, img_encoding.shape[1])

    return img_encoding.reshape((1, 2048))


def greedySearch(photo):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = [wordtoix[w] for w in in_text.split() if w in wordtoix]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = captioning_model.predict([photo, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = ixtoword[yhat]
        in_text += ' ' + word
        if word == 'endseq':
            break
    final = in_text.split()
    final = final[1:-1]
    final = ' '.join(final)
    return final


@app.post("/predict-caption")
async def predict_caption(file: UploadFile = File(...)):
    # Save the uploaded file
    with open("temp_image.jpg", "wb") as temp_image:
        temp_image.write(file.file.read())

    # Get the encoding for the new image
    new_image_encoding = encode_image("temp_image.jpg")

    # Generate caption
    predicted_caption = greedySearch(new_image_encoding.reshape((1, 2048)))

    # Remove the temporary image file
    import os
    os.remove("temp_image.jpg")

    return JSONResponse(content={"predicted_caption": predicted_caption})
