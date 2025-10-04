import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def analisar_outliers(df: pd.DataFrame) -> str:
    """Análise direta de outliers sem agentes"""
    numeric_cols = df.select_dtypes(include=['number']).columns
    
    resultado = "=== ANÁLISE DE VALORES ATÍPICOS (OUTLIERS) ===\n\n"
    
    for col in numeric_cols[:6]:  # Primeiras 6 colunas
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        
        outliers = df[(df[col] < lower) | (df[col] > upper)]
        pct = len(outliers) / len(df) * 100
        
        resultado += f"**{col}**:\n"
        resultado += f"  - Outliers detectados: {len(outliers):,} ({pct:.2f}%)\n"
        resultado += f"  - Limites: [{lower:.2f}, {upper:.2f}]\n\n"
    
    # Gerar gráfico
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.ravel()
    
    for i, col in enumerate(numeric_cols[:4]):
        df[col].plot(kind='box', ax=axes[i])
        axes[i].set_title(f'Boxplot: {col}')
        axes[i].set_ylabel('Valor')
    
    plt.tight_layout()
    plt.show()
    
    return resultado

def analisar_tipos_dados(df: pd.DataFrame) -> str:
    """Análise de tipos de dados"""
    resultado = "=== TIPOS DE DADOS ===\n\n"
    
    numeric = df.select_dtypes(include=['number']).columns.tolist()
    categoric = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    resultado += f"**Variáveis Numéricas ({len(numeric)}):**\n"
    for col in numeric:
        resultado += f"  - {col}: {df[col].dtype}\n"
    
    resultado += f"\n**Variáveis Categóricas ({len(categoric)}):**\n"
    for col in categoric:
        resultado += f"  - {col}: {df[col].dtype}\n"
    
    return resultado

# Dicionário de funções diretas
FUNCOES_EDA = {
    "outliers": analisar_outliers,
    "atípicos": analisar_outliers,
    "tipos": analisar_tipos_dados,
    "tipos de dados": analisar_tipos_dados,
}