from fastapi import FastAPI, File, UploadFile
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import io
import tensorflow as tf
import cv2
import sys
import os
sys.path.append(os.path.abspath('../../'))
from gan.data.utils import Utils

app = FastAPI()

# Permitir CORS para testing desde cualquier origen
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Configuración ---
MODEL_PATH = "../../gan/models/trained_512_generator.keras"
TARGET_SIZE = 512  # o 1024 si entrenaste a mayor resolución

# --- Cargar el modelo ---
print("🔄 Cargando modelo...")
generator = tf.keras.models.load_model(MODEL_PATH)
print("✅ Modelo cargado.")

# --- Endpoint para generar imagen ---
@app.post("/predict")
def predict(image: UploadFile = File(...)):
    image_bytes = image.file.read()
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    shape = img.shape
    # Preprocesar
    img = Utils.resize_with_padding_or_crop(img, TARGET_SIZE)
    input_img = (img.astype(np.float32) / 127.5) - 1.0
    input_tensor = np.expand_dims(input_img, axis=0)

    # Generar
    print("🎨 Generando estilo...")
    prediction = generator(input_tensor, training=False)
    output_img = prediction.numpy()[0]
    output_img = ((output_img + 1.0) * 127.5).astype(np.uint8)
    output_img = Utils.resize_with_crop(output_img, shape,TARGET_SIZE)

    # Codificar como PNG
    _, encoded_img = cv2.imencode('.png', output_img)
    return Response(content=encoded_img.tobytes(), media_type="image/png")