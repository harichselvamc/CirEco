from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import io
from PIL import Image

app = FastAPI()

# Add CORS middleware to allow requests from the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the trained model
model_path = 'model_small.h5'
model = load_model(model_path)

# Define the class names (ensure these are in the same order as during training)
class_names = ['organic', 'recycle']  # Replace with your actual class names

# Function to preprocess the image
def preprocess_image(img: Image.Image, img_width: int, img_height: int):
    img = img.resize((img_width, img_height))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

@app.post("/predict/")
async def predict_image(file: UploadFile = File(...)):
    img = Image.open(io.BytesIO(await file.read()))
    img_array = preprocess_image(img, img_width=80, img_height=45)  # Ensure these match your model's input size
    prediction = model.predict(img_array)
    predicted_class_index = np.argmax(prediction)
    predicted_class = class_names[predicted_class_index]
    return JSONResponse({"predicted_class": predicted_class})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
