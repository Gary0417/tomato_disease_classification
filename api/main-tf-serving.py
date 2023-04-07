import uvicorn 
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile
from io import BytesIO
from PIL import Image


app = FastAPI()

MODEL = tf.keras.models.load_model("../saved_models/2")

# List of class names for prediction
CLASS_NAMES = ['Tomato_Bacterial_spot',
               'Tomato_Early_blight',
               'Tomato_Late_blight',
               'Tomato_Leaf_Mold',
               'Tomato_Septoria_leaf_spot',
               'Tomato_Spider_mites_Two_spotted_spider_mite',
               'Tomato__Target_Spot',
               'Tomato__Tomato_YellowLeaf__Curl_Virus',
               'Tomato__Tomato_mosaic_virus',
               'Tomato_healthy']

# Utility function to read file as image and convert to numpy array
def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    # Read the uploaded file as an image
    image = read_file_as_image(await file.read())

    # Expand dimensions to create a batch of size 1
    img_batch = np.expand_dims(image, 0)
    
    predictions = MODEL.predict(img_batch)

    # Get the predicted class with highest probability
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    
    confidence = np.max(predictions[0])
    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }

if __name__ == '__main__':
    uvicorn.run(app, host = 'localhost', port = 8000)