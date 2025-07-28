import os
import numpy as np
import cv2
from glob import glob
import tensorflow as tf
from data.utils import Utils

class Dataset:
    def __init__(self,input_dir,target_dir,target_size,batch_size=2):
        self.input_dir = input_dir
        self.target_dir = target_dir
        self.target_size = target_size
        self.batch_size = batch_size
        self.dataset = self.load_dataset()

    # --- Cargar par de imágenes emparejadas ---
    def load_image_pair(self,input_path, target_path):
        input_img = cv2.imread(input_path)
        target_img = cv2.imread(target_path)

        input_img = Utils.resize_with_padding_or_crop(input_img, self.target_size)
        target_img = Utils.resize_with_padding_or_crop(target_img, self.target_size)

        input_img = (input_img.astype(np.float32) / 127.5) - 1.0
        target_img = (target_img.astype(np.float32) / 127.5) - 1.0

        return input_img, target_img

    # --- Dataset ---
    def load_dataset(self):
        input_files = sorted(glob(os.path.join(self.input_dir, "*")))
        target_files = sorted(glob(os.path.join(self.target_dir, "*")))
        dataset = []
        for in_path, tgt_path in zip(input_files, target_files):
            inp, tgt = self.load_image_pair(in_path, tgt_path)
            dataset.append((inp, tgt))

        inputs, targets = zip(*dataset)
        inputs = np.array(inputs)
        targets = np.array(targets)
        return tf.data.Dataset.from_tensor_slices((inputs, targets)).shuffle(1000).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
