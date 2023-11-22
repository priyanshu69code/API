# %%
import pickle
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing import image
import numpy as np
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from keras.applications.inception_v3 import InceptionV3
from keras.models import Model

# %%
max_length = 33

# %%


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


# %%

# Load the model from the saved file
# Load the inception v3 model
model = InceptionV3(weights='imagenet')


# Create a new model, by removing the last layer (output layer) from the inception v3
model_new = Model(model.input, model.layers[-2].output)

# %%
# Function to encode a given image into a vector of size (2048, )


def encode(image):
    image = preprocess(image)  # preprocess the image
    fea_vec = model_new.predict(image)  # Get the encoding vector for the image
    # reshape from (1, 2048) to (2048, )
    fea_vec = np.reshape(fea_vec, fea_vec.shape[1])
    return fea_vec


# %%

# Load the dictionary from the pickle file
with open("wordtoix.pkl", "rb") as encoded_pickle:
    wordtoix = pickle.load(encoded_pickle)

# Now, 'loaded_wordtoix' contains the dictionary loaded from the file


# %%

# Load the dictionary from the pickle file
with open("ixtoword.pkl", "rb") as encoded_pickle:
    ixtoword = pickle.load(encoded_pickle)

# Now, 'loaded_ixtoword' contains the dictionary loaded from the file


# %%
model = load_model("main_model.h5")

# %%


def greedySearch(photo):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = [wordtoix[w] for w in in_text.split() if w in wordtoix]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([photo, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = ixtoword[yhat]
        in_text += ' ' + word
        if word == 'endseq':
            break
    final = in_text.split()
    final = final[1:-1]
    final = ' '.join(final)
    return final


# %%


def encode_image(image_path):
    # Load and preprocess the image
    img = image.load_img(image_path, target_size=(299, 299))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Get the image encoding using the pre-trained InceptionV3 model
    img_encoding = model_new.predict(img_array)
    img_encoding = np.reshape(img_encoding, img_encoding.shape[1])

    return img_encoding.reshape((1, 2048))


# %%
# Replace 'path/to/your/image.jpg' with the actual path to your new image
new_image_path = 'test.jpg'

# Get the encoding for the new image
new_image_encoding = encode_image(new_image_path)

# Now you can use the new_image_encoding for caption generation
predicted_caption = greedySearch(new_image_encoding)
print("Predicted Caption:", predicted_caption)
