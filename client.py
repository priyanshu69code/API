import requests
from pathlib import Path

# Replace this with the actual path to your image file
image_path = "test.jpg"

# API endpoint URL
api_url = "http://127.0.0.1:8000/predict-caption"

# Open the image file
with open(image_path, "rb") as img_file:
    # Prepare the files dictionary for the request
    files = {"file": (Path(image_path).name, img_file, "image/jpeg")}

    # Make the API request
    response = requests.post(api_url, files=files)

# Check if the request was successful (status code 200)
if response.status_code == 200:
    # Parse the JSON response
    data = response.json()
    predicted_caption = data.get("predicted_caption")

    # Display the predicted caption
    print("Predicted Caption:", predicted_caption)
else:
    # Display an error message if the request was not successful
    print("Error:", response.text)
