#src\data\bases.py
import kagglehub
import os
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

PASTA_LOCAL = os.getenv('PASTA_LOCAL', 'data/base')

# Download das Bases
def download_base():
    """Executa o download da base de dados e remove os duplicados"""
    #Definir o local do arquivo
    caminho_local_csv = os.path.join(PASTA_LOCAL, 'creditcard.csv')

    # Ferifica se a pasta existe e caso não a cria
    os.makedirs(PASTA_LOCAL, exist_ok=True)

    # Ferificar se o arquivo existe na pasta local
    if os.path.exists(caminho_local_csv):
        print('Arquivo existe localmente. Carregando...')
        df = pd.read_csv(caminho_local_csv)
    else:
        print('Arquivo não encontrado. Baixando do Kagge')
        # Download Base
        download = kagglehub.dataset_download("mlg-ulb/creditcardfraud")

        caminho_csv = os.path.join(download, "creditcard.csv")
        df = pd.read_csv(caminho_csv)

        # Salvar na pasta local para utilização futira
        df.to_csv(caminho_local_csv, index=False)
        print('Download concluído e salvo na pasta localmente')

    # Padronizar as colunas
    df.columns = df.columns.str.strip().str.upper()


    #Tratar os dados
    df.drop_duplicates(inplace=True)

    # Remover colunas vazias
    df = df.dropna(axis=1, how="all")

    return df

