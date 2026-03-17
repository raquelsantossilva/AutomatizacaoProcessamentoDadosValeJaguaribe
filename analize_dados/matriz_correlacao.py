import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os

caminho = r"C:\Users\12bea\OneDrive\Documentos\SBBD_algoritms\dados_funceme_jaguaribe"

arquivos_dados = glob.glob(os.path.join(caminho , "station_*_data.csv"))
lista = []
for ar in arquivos_dados:
    df = pd.read_csv(ar)
    lista.append(df)

df_par = pd.concat(lista , ignore_index= True)
pd.set_option('display.max_columns', None)
df_par = df_par.drop("data", axis=1)
corr = df_par.corr()["Temperatura do Ar a 2m"]
print(corr)
