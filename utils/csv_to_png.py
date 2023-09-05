# Link para fazer o download do dataset:
# https://www.kaggle.com/datasets/deadskull7/fer2013?select=fer2013.csv
# Esse script converte o arquivo csv para imagens png
# O arquivo csv contém 35887 imagens de 48x48 pixels
# Colocar o arquivo csv em uma pasta chamada 'datasets_raw'

# Bibliotecas necessárias
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.utils import shuffle
from tqdm import tqdm

# Definindo os diretórios
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_NAME = "fer2013"
DATASET_DIR = BASE_DIR / "datasets" / DATASET_NAME
DATASET_RAW_DIR = BASE_DIR / "datasets_raw"
DATASET_DIR.mkdir(parents=True, exist_ok=True)
DATASET_RAW_DIR.mkdir(parents=True, exist_ok=True)

# Definindo as constantes
TRAIN_PERCENT = 0.80
IMG_SIZE = 48

outer_folder_names = [
    "train",
    "validation",
]
inner_folder_names = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "sad",
    "surprise",
    "neutral",
]

# Criando as pastas
for outer_folder_name in outer_folder_names:
    for inner_folder_name in inner_folder_names:
        folder_path = DATASET_DIR / outer_folder_name / inner_folder_name
        folder_path.mkdir(parents=True, exist_ok=True)

# Lendo o arquivo csv
df = pd.read_csv(DATASET_RAW_DIR / f"{DATASET_NAME}.csv")
df = shuffle(df)
mat = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)

# Salvando as imagens
print("Salvando as imagens...")

for i in tqdm(range(len(df))):
    txt = df["pixels"][i]
    words = txt.split()

    for j in range(2304):
        xind = j // IMG_SIZE
        yind = j % IMG_SIZE
        mat[xind][yind] = int(words[j])
    img = Image.fromarray(mat)

    if i < TRAIN_PERCENT * len(df):
        img.save(DATASET_DIR / "train" / inner_folder_names[df["emotion"][i]] / f"{i}.png")
    else:
        img.save(DATASET_DIR / "validation" / inner_folder_names[df["emotion"][i]] / f"{i}.png")

print("Imagens salvas com sucesso!")
