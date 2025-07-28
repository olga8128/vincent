import cv2
import numpy as np
import os

class Utils:
    
    # --- Preprocesamiento avanzado (padding o crop) ---
    @classmethod
    def resize_with_padding_or_crop(self,img, target_size):
        w, h, _ = img.shape
        if h>w:
            k=target_size/h
            new_w = round(k*w)
            img = cv2.resize(img, (target_size, new_w))
            left = (target_size - new_w) // 2
            right = target_size - new_w - left
            img = cv2.copyMakeBorder(img, left, right, 0, 0, cv2.BORDER_CONSTANT, value=[255, 255, 255])
        else:
            k = target_size/w
            new_h=round(k*h)
            img = cv2.resize(img, (new_h, target_size))
            top = (target_size - new_h) // 2 
            bottom = target_size - new_h - top
            img = cv2.copyMakeBorder(img, 0, 0, top, bottom, cv2.BORDER_CONSTANT, value=[255, 255, 255])

        return img
    
    # --- Postprocesamiento avanzado (padding o crop) ---
    @classmethod
    def resize_with_crop(self,img, size,target_size):
        h,w,_ = size
        if h>w:
            new_w=round((w/h)*target_size)
            left = (target_size - new_w) // 2
            right = target_size - new_w - left
            img = img[0:target_size,left:target_size-right, ]
        else:
            new_h=round((h/w)*target_size)
            top = (target_size - new_h) // 2 
            bottom = target_size - new_h - top
            img = img[top:target_size-bottom,0:target_size]
        return img

    
    @classmethod
    def cuartear_imagen_segmentada(self,img,k=10):
        # Leer y suavizar la imagen para evitar detalles finos
        img = self.cargar_y_preparar_imagen(img)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Preparar para K-means
        Z = img_rgb.reshape((-1, 3)).astype(np.float32)

        # K-means clustering
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1.0)
        _, labels, centers = cv2.kmeans(Z, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        centers = np.uint8(centers)
        segmented_img = centers[labels.flatten()].reshape(img_rgb.shape)

        # Aplicar morfología para suavizar áreas de color en la imagen final
        kernel = np.ones((9, 9), np.uint8)  # Aumenta tamaño para mayor suavidad
        segmented_morph = np.zeros_like(segmented_img)

        for c in range(3):  # Aplicar por canal R, G, B
            segmented_morph[..., c] = cv2.morphologyEx(segmented_img[..., c], cv2.MORPH_OPEN, kernel)
            segmented_morph[..., c] = cv2.morphologyEx(segmented_morph[..., c], cv2.MORPH_CLOSE, kernel)

        # Usar la imagen morfológicamente suavizada como resultado final
        segmented_img = segmented_morph

        # Suavizar bordes entre regiones
        segmented_img = self.cargar_y_preparar_imagen(segmented_img)

        return segmented_img
    
    @classmethod
    def cargar_y_preparar_imagen(self,img, blur_ksize=7):

        if img is None:
            raise ValueError("Error al cargar la imagen. Verifica la ruta.")

        # Si viene con dimensión extra, quitarla
        if len(img.shape) == 4 and img.shape[0] == 1:
            img = img[0]

        # Asegurar tipo
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)

        # Convertir si es grayscale
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        # Verificar y aplicar blur
        if len(img.shape) == 3 and img.shape[2] == 3:
            if blur_ksize % 2 == 1:
                img = cv2.medianBlur(img, blur_ksize)
            else:
                raise ValueError("El valor de ksize debe ser impar.")
        else:
            raise ValueError(f"Dimensiones incompatibles para medianBlur: {img.shape}")

        return img

    @classmethod
    def crear_imagenes_input(self,input_dir,target_dir):
        directory = os.fsencode(input_dir)
        for file in os.listdir(directory):
            filename = os.fsdecode(file)
            if filename.endswith(".jpg") or filename.endswith(".png"): 
                img = cv2.imread(input_dir+"/"+filename)
                img = self.cuartear_imagen_segmentada(img,k=5)
                self.save_image(img,filename,target_dir)
        print("☑️  Creación de cuarteados completada.")

    @classmethod
    def save_image(self,img,filename,dir):
        output_seg_path = dir+"/"+self.get_cuarteado_name(filename)
        cv2.imwrite(output_seg_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    @classmethod
    def get_cuarteado_name(self,filename):
        return f"cuarteado_{filename}"