from gan_ia import GanIA
import os

#Generar imagen con la GAN entrenada
gan_ia = GanIA(512)
directory = os.fsencode(GanIA.GENERATED_INPUT_DIR)
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    gan_ia.generate(filename)
    