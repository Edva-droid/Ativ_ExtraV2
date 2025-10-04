# utils.py
import pandas as pd
import hashlib

def gerar_hash_df_leve(df: pd.DataFrame, n_amostra: int = 5) -> str:
    """
    Cria um hash leve baseado nas colunas, tamanho do DF e
    algumas linhas de amostra.
    """
    amostra = df.head(n_amostra).to_csv(index=False)
    base = (
        str(df.shape) +            # linhas e colunas
        "".join(df.columns) +      # nomes das colunas
        amostra                     # primeiras n linhas
    )
    return hashlib.md5(base.encode()).hexdigest()


def analise_dados(df, pergunta: str) -> str:
    """
    Tool que responde perguntas matemáticas exatas sobre o DataFrame.
    """
    try:
        pergunta_lower = pergunta.lower()
        
        if "maior valor" in pergunta_lower:
            col = pergunta.split()[-1]  # extrai última palavra como coluna
            if col in df.columns:
                return f"O maior valor da coluna {col} é {df[col].max()}"
        
        if "menor valor" in pergunta_lower:
            col = pergunta.split()[-1]
            if col in df.columns:
                return f"O menor valor da coluna {col} é {df[col].min()}"

        if "média" in pergunta_lower:
            col = pergunta.split()[-1]
            if col in df.columns:
                return f"A média da coluna {col} é {df[col].mean()}"

        if "mediana" in pergunta_lower:
            col = pergunta.split()[-1]
            if col in df.columns:
                return f"A mediana da coluna {col} é {df[col].median()}"
        
        return "❓ Não consegui identificar uma operação matemática clara."

    except Exception as e:
        return f"⚠️ Erro ao processar: {e}"
