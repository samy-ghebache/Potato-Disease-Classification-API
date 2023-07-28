from fastapi import FastAPI,File,UploadFile
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
from fastapi.middleware.cors import CORSMiddleware

app=FastAPI()

MODEL=tf.keras.models.load_model("./Potato-Disease.h5")

CLASSES=['Early blight', 'Late blight', 'healthy']


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post('/predict')
async def predict(file: UploadFile=File(...)):
    bytes=await file.read()
    image=np.array(Image.open(BytesIO(bytes)))
    image=np.expand_dims(image,0)
    predict=MODEL.predict(image)[0]
    class_name=CLASSES[np.argmax(predict)]
    prediction=round(100*(np.max(predict)),2)
    return {"class":class_name,"prediction":prediction}
    