from data.utils import Utils
import tensorflow as tf
import cv2
import numpy as np

class Generator:
    def __init__(self,target_size,model_path,generator=None):
        print("🔄 Cargando modelo...")
        if generator:
            self.generator = generator
        else:
            self.generator = tf.keras.models.load_model(model_path)
        self.target_size = target_size

    # --- Cargar imagen de entrada ---
    def load_input_image(self,path):
        img = cv2.imread(path)
        img = Utils.resize_with_padding_or_crop(img, self.target_size)
        img = (img.astype(np.float32) / 127.5) - 1.0  # Normalizar a [-1, 1]
        img = np.expand_dims(img, axis=0)
        return img

    # --- Guardar imagen generada ---
    def save_output_image(self,img, path):
        img = (img[0] + 1.0) * 127.5  # Volver a [0, 255]
        img = np.clip(img, 0, 255).astype(np.uint8)
        cv2.imwrite(path, img)

    # --- Cargar modelo y aplicar ---
    def apply_style(self,input_path,output_path):
        print(f"📷 Procesando imagen: {input_path}")
        input_image = self.load_input_image(input_path)
        print("🎨 Generando estilo...")
        prediction = self.generator(input_image, training=False)
        print(f"💾 Guardando imagen generada: {output_path}")
        self.save_output_image(prediction.numpy(), output_path)
        print("✅ Completado.")