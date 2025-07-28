from neural.model import Model
from neural.generator import Generator
from data.dataset import Dataset
from data.get_data import GetDataWikiart
from data.utils import Utils
import os
import cv2
import numpy as np

class GanIA:
    INPUT_DIR="images/training/input"  
    TARGET_DIR="images/training/target"
    MODELS_DIR="models"
    GENERATED_INPUT_DIR="images/generated/input/"
    GENERATED_OUTPUT_DIR="images/generated/output/"
    DEFAULT_MODEL="pix2pix"
    ARTIST="Vincent Van Gogh"
    MAX_IMAGES=10

    def __init__(self,target_size,new_model_name=DEFAULT_MODEL,saved_gen_name=Model.GEN_NAME+DEFAULT_MODEL,saved_dis_name=Model.DIS_NAME+DEFAULT_MODEL):
        self.create_dirs()
        self.model_name=new_model_name
        self.target_size = target_size
        saved_gen_path = self.get_model_dir(saved_gen_name,target_size,gen=True)
        saved_dis_path = self.get_model_dir(saved_dis_name,target_size,gen=False)
        new_model_path=self.get_model_dir(new_model_name,target_size,gen=True)
        if os.path.exists(saved_gen_path) and os.path.exists(saved_dis_path):
            self.model = Model(self.target_size,new_model_path,saved_gen_path,saved_dis_path)
        elif os.path.exists(saved_gen_path):
            self.model = Model(self.target_size,new_model_path,saved_gen_path)
        else:
            self.model = Model(self.target_size,new_model_path)

    def create_dirs(self):
        os.makedirs(self.INPUT_DIR, exist_ok=True)
        os.makedirs(self.TARGET_DIR, exist_ok=True)
        os.makedirs(self.MODELS_DIR, exist_ok=True)
        os.makedirs(self.GENERATED_INPUT_DIR, exist_ok=True)
        os.makedirs(self.GENERATED_OUTPUT_DIR, exist_ok=True)

    def train(self,epochs,dataset_number=None):
        if dataset_number:
            set_folder=f'/dataset_{dataset_number:03d}'
            dataset=Dataset(self.INPUT_DIR+set_folder,self.TARGET_DIR+set_folder,self.target_size)
        else:    
            dataset=Dataset(self.INPUT_DIR,self.TARGET_DIR,self.target_size)
        self.model.train(dataset,epochs)

    def generate(self,filename):
        gen = self.model.generator
        generator=Generator(self.target_size,self.get_model_dir(self.model_name,self.target_size,True),gen)
        generator.apply_style(self.GENERATED_INPUT_DIR+filename,self.GENERATED_OUTPUT_DIR+filename)

    def get_model_dir(self,model,target_size,gen):
        if gen:
            return self.MODELS_DIR+f"/{model}_{target_size}{Model.GEN_NAME}.keras"
        else:
            return self.MODELS_DIR+f"/{model}_{target_size}{Model.DIS_NAME}.keras"
    
    def get_train_data(self,first_page,num_pages):
        wikiart=GetDataWikiart(self.TARGET_DIR,self.ARTIST)
        wikiart.download_pages(self.MAX_IMAGES,first_page,first_page+num_pages)
        Utils.crear_imagenes_input(self.TARGET_DIR,self.INPUT_DIR)
    
def main():
    #Crear red neuronal GAN de imagenes 512x512
    gan_ia = GanIA(target_size=512)
    #Descargar datos de entrenamiento
    gan_ia.get_train_data(first_page=1,num_pages=1)
    #Entrenar
    gan_ia.train(epochs=20)

if __name__=="__main__":
    main()
    
    
