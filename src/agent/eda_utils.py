import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from typing import Dict, List, Any, Optional
import warnings
warnings.filterwarnings('ignore')

def analise_eda_completa(
    df: pd.DataFrame,
    salvar_graficos: bool = False,  # Para Streamlit, melhor não salvar
    pasta_graficos: str = "graficos_eda",
    max_insights: int = 10,
    target_column: Optional[str] = None
) -> Dict[str, Any]:
    """
    Realiza EDA completa para qualquer DataFrame, otimizada para LLM
    
    Args:
        df: DataFrame para análise
        salvar_graficos: Se deve salvar gráficos em arquivos
        pasta_graficos: Pasta para salvar gráficos
        max_insights: Máximo de insights a gerar
        target_column: Coluna target para análises específicas
    
    Returns:
        dict: Resultados estruturados da análise
    """
    if salvar_graficos and not os.path.exists(pasta_graficos):
        os.makedirs(pasta_graficos)
   
    resultados = {}
    insights = []
    
    # ================================
    # 1. INFORMAÇÕES BÁSICAS
    # ================================
    info_basica = {
        "shape": df.shape,
        "colunas": df.columns.tolist(),
        "tipos_dados": df.dtypes.astype(str).to_dict(),
        "memoria_mb": round(df.memory_usage(deep=True).sum() / 1024**2, 2),
        "valores_nulos": df.isnull().sum().to_dict(),
        "valores_duplicados": int(df.duplicated().sum())
    }
    resultados["info_basica"] = info_basica
    
    # ================================
    # 2. ANÁLISE POR TIPO DE VARIÁVEL
    # ================================
    colunas_numericas = df.select_dtypes(include=['float64', 'int64', 'int32', 'float32']).columns.tolist()
    colunas_categoricas = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Estatísticas descritivas
    if colunas_numericas:
        estatisticas = df[colunas_numericas].describe().T
        resultados["estatisticas_numericas"] = estatisticas.to_dict()
        
        # Análise de distribuição
        distribuicao = {}
        for col in colunas_numericas:
            distribuicao[col] = {
                "skewness": float(df[col].skew()),
                "kurtosis": float(df[col].kurtosis()),
                "outliers_iqr": _detectar_outliers_iqr(df[col]),
                "zeros": int((df[col] == 0).sum()),
                "negativos": int((df[col] < 0).sum())
            }
        resultados["distribuicao"] = distribuicao
    
    # Análise categórica
    if colunas_categoricas:
        categoricas_info = {}
        for col in colunas_categoricas:
            value_counts = df[col].value_counts()
            categoricas_info[col] = {
                "valores_unicos": int(df[col].nunique()),
                "mais_frequente": str(value_counts.index[0]),
                "freq_mais_frequente": int(value_counts.iloc[0]),
                "menos_frequente": str(value_counts.index[-1]),
                "freq_menos_frequente": int(value_counts.iloc[-1]),
                "cardinalidade": float(df[col].nunique() / len(df))
            }
        resultados["categoricas_info"] = categoricas_info
    
    # ================================
    # 3. CORRELAÇÕES
    # ================================
    if len(colunas_numericas) >= 2:
        corr = df[colunas_numericas].corr()
        resultados["correlacao"] = corr.to_dict()
        
        # Correlações fortes
        correlacoes_fortes = []
        for i in range(len(corr.columns)):
            for j in range(i+1, len(corr.columns)):
                corr_val = corr.iloc[i, j]
                if abs(corr_val) > 0.5:
                    correlacoes_fortes.append({
                        "var1": corr.columns[i],
                        "var2": corr.columns[j],
                        "correlacao": float(corr_val)
                    })
        resultados["correlacoes_fortes"] = correlacoes_fortes
    
    # ================================
    # 4. ANÁLISE DA VARIÁVEL TARGET
    # ================================
    if target_column and target_column in df.columns:
        target_info = _analisar_target(df, target_column, colunas_numericas)
        resultados["analise_target"] = target_info
    
    # ================================
    # 5. GERAÇÃO DE INSIGHTS
    # ================================
    insights = _gerar_insights_automaticos(df, colunas_numericas, colunas_categoricas, 
                                          resultados, target_column, max_insights)
    
    # ================================
    # 6. GRÁFICOS (para Streamlit)
    # ================================
    graficos_gerados = []
    
    # # Histogramas e Boxplots para numéricas
    # for col in colunas_numericas[:6]:  # Limite para não sobrecarregar
    #     try:
    #         fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            
    #         # Histograma
    #         df[col].hist(bins=30, ax=ax1, alpha=0.7)
    #         ax1.set_title(f'Distribuição de {col}')
    #         ax1.set_xlabel(col)
    #         ax1.set_ylabel('Frequência')
            
    #         # Boxplot
    #         df[col].plot(kind='box', ax=ax2)
    #         ax2.set_title(f'Boxplot de {col}')
    #         ax2.set_ylabel(col)
            
    #         plt.tight_layout()
            
    #         if salvar_graficos:
    #             plt.savefig(f"{pasta_graficos}/dist_{col}.png", dpi=100, bbox_inches='tight')
            
    #         graficos_gerados.append(f"Distribuição de {col}")
    #         plt.show()  # Para Streamlit
    #         plt.close()
            
    #     except Exception as e:
    #         print(f"Erro ao criar gráfico para {col}: {e}")
    
    # # Heatmap de correlação
    # if len(colunas_numericas) >= 2:
    #     try:
    #         plt.figure(figsize=(10, 8))
    #         mask = np.triu(np.ones_like(corr, dtype=bool))
    #         sns.heatmap(corr, mask=mask, annot=True, cmap='RdBu_r', center=0,
    #                    square=True, fmt='.2f')
    #         plt.title('Matriz de Correlação')
    #         plt.tight_layout()
            
    #         if salvar_graficos:
    #             plt.savefig(f"{pasta_graficos}/correlacao.png", dpi=100, bbox_inches='tight')
            
    #         graficos_gerados.append("Matriz de Correlação")
    #         plt.show()  # Para Streamlit
    #         plt.close()
            
    #     except Exception as e:
    #         print(f"Erro ao criar heatmap: {e}")
    
    # ================================
    # 7. RESUMO EXECUTIVO PARA LLM
    # ================================
    resumo = {
        "dataset_overview": {
            "linhas": df.shape[0],
            "colunas": df.shape[1],
            "variaveis_numericas": colunas_numericas,
            "variaveis_categoricas": colunas_categoricas,
            "qtde_numericas": len(colunas_numericas),
            "qtde_categoricas": len(colunas_categoricas),
            "memoria_mb": info_basica["memoria_mb"],
            "completude": round((1 - df.isnull().sum().sum() / df.size) * 100, 1)
        },
        "principais_achados": insights,
        "qualidade_dados": {
            "valores_nulos": df.isnull().sum().sum(),
            "duplicatas": info_basica["valores_duplicados"],
            "colunas_constantes": int((df.nunique() == 1).sum())
        },
        "graficos_gerados": graficos_gerados
    }
    
    if target_column and "analise_target" in resultados:
        resumo["target_insights"] = resultados["analise_target"]["insights"]
    
    resultados["resumo_executivo"] = resumo
    resultados["insights"] = insights
    
    return resultados


def _detectar_outliers_iqr(serie: pd.Series) -> Dict[str, int]:
    """Detecta outliers usando método IQR"""
    Q1 = serie.quantile(0.25)
    Q3 = serie.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = serie[(serie < lower_bound) | (serie > upper_bound)]
    
    return {
        "quantidade": len(outliers),
        "percentual": round((len(outliers) / len(serie)) * 100, 2),
        "limite_inferior": float(lower_bound),
        "limite_superior": float(upper_bound)
    }


def _analisar_target(df: pd.DataFrame, target_col: str, colunas_numericas: List[str]) -> Dict[str, Any]:
    """Análise específica da variável target"""
    target_info = {
        "tipo": str(df[target_col].dtype),
        "valores_unicos": int(df[target_col].nunique()),
        "distribuicao": df[target_col].value_counts().to_dict(),
        "insights": []
    }
    
    # Se for binária (classificação)
    if df[target_col].nunique() == 2:
        dist = df[target_col].value_counts()
        ratio = dist.min() / dist.max()
        target_info["balanceamento"] = {
            "ratio": float(ratio),
            "balanceada": ratio > 0.3
        }
        
        if ratio < 0.1:
            target_info["insights"].append(f"Dataset extremamente desbalanceado (ratio: {ratio:.2f})")
        elif ratio < 0.3:
            target_info["insights"].append(f"Dataset desbalanceado (ratio: {ratio:.2f})")
    
    # Correlação com variáveis numéricas
    if colunas_numericas and df[target_col].dtype in ['int64', 'float64']:
        correlacoes = df[colunas_numericas + [target_col]].corr()[target_col].abs().sort_values(ascending=False)
        correlacoes = correlacoes.drop(target_col)  # Remove self-correlation
        
        target_info["correlacoes_com_features"] = correlacoes.head(5).to_dict()
        
        if correlacoes.iloc[0] > 0.7:
            target_info["insights"].append(f"Correlação muito forte com {correlacoes.index[0]} ({correlacoes.iloc[0]:.2f})")
    
    return target_info


def _gerar_insights_automaticos(df: pd.DataFrame, numericas: List[str], categoricas: List[str], 
                               resultados: Dict, target_col: Optional[str], max_insights: int) -> List[str]:
    """Gera insights automáticos baseados na análise"""
    insights = []
    
    # Insights sobre distribuições
    if numericas and "distribuicao" in resultados:
        for col, info in resultados["distribuicao"].items():
            skew = info["skewness"]
            if abs(skew) > 1:
                direcao = "direita" if skew > 0 else "esquerda"
                insights.append(f"'{col}' tem distribuição assimétrica à {direcao} (skew={skew:.2f})")
            
            if info["outliers_iqr"]["percentual"] > 5:
                insights.append(f"'{col}' tem {info['outliers_iqr']['percentual']:.1f}% de outliers")
    
    # Insights sobre correlações
    if "correlacoes_fortes" in resultados and resultados["correlacoes_fortes"]:
        for corr in resultados["correlacoes_fortes"][:3]:  # Top 3
            tipo = "positiva" if corr["correlacao"] > 0 else "negativa"
            insights.append(f"Correlação {tipo} forte entre '{corr['var1']}' e '{corr['var2']}' ({corr['correlacao']:.2f})")
    
    # Insights sobre variáveis categóricas
    if categoricas and "categoricas_info" in resultados:
        for col, info in resultados["categoricas_info"].items():
            if info["cardinalidade"] > 0.9:
                insights.append(f"'{col}' tem alta cardinalidade ({info['valores_unicos']} valores únicos)")
            elif info["cardinalidade"] < 0.1:
                insights.append(f"'{col}' tem baixa variabilidade (apenas {info['valores_unicos']} valores únicos)")
    
    # Insights sobre qualidade dos dados
    valores_nulos = df.isnull().sum().sum()
    if valores_nulos > 0:
        perc_nulos = (valores_nulos / df.size) * 100
        if perc_nulos > 10:
            insights.append(f"Dataset tem {perc_nulos:.1f}% de valores nulos")
    
    duplicatas = df.duplicated().sum()
    if duplicatas > 0:
        perc_dup = (duplicatas / len(df)) * 100
        insights.append(f"Dataset tem {duplicatas} linhas duplicadas ({perc_dup:.1f}%)")
    
    # Insights sobre target
    if target_col and target_col in df.columns:
        if df[target_col].nunique() == 2:
            dist = df[target_col].value_counts()
            ratio = dist.min() / dist.max()
            if ratio < 0.3:
                insights.append(f"Variável target '{target_col}' está desbalanceada (ratio: {ratio:.2f})")
    
    return insights[:max_insights]


# Função wrapper para integração com o sistema atual
def executar_eda_completa(df: pd.DataFrame, target_column: Optional[str] = None) -> str:
    """
    Função wrapper para usar no sistema de agentes
    Retorna resultado formatado para o LLM
    """
    try:
        resultados = analise_eda_completa(df, salvar_graficos=False, target_column=target_column)
        resumo = resultados["resumo_executivo"]
        
        # Formatar resposta para o LLM
        resposta = f"""
# ANÁLISE EXPLORATÓRIA COMPLETA

## 📊 Visão Geral do Dataset
- **Dimensões:** {resumo['dataset_overview']['linhas']:,} linhas × {resumo['dataset_overview']['colunas']} colunas
- **Variáveis Numéricas:** {resumo['dataset_overview']['qtde_numericas']} ({', '.join(resumo['dataset_overview']['variaveis_numericas'][:5])})
- **Variáveis Categóricas:** {resumo['dataset_overview']['qtde_categoricas']} ({', '.join(resumo['dataset_overview']['variaveis_categoricas'][:5])})
- **Completude dos Dados:** {resumo['dataset_overview']['completude']}%
- **Uso de Memória:** {resumo['dataset_overview']['memoria_mb']} MB

## 🔍 Principais Descobertas
"""
        
        for i, insight in enumerate(resumo["principais_achados"], 1):
            resposta += f"{i}. {insight}\n"
        
        resposta += f"""
## 📈 Qualidade dos Dados
- **Valores Nulos:** {resumo['qualidade_dados']['valores_nulos']:,}
- **Linhas Duplicadas:** {resumo['qualidade_dados']['duplicatas']:,}
- **Colunas Constantes:** {resumo['qualidade_dados']['colunas_constantes']}

## 📊 Gráficos Gerados
"""
        for grafico in resumo["graficos_gerados"]:
            resposta += f"- {grafico}\n"
        
        if target_column and "target_insights" in resumo:
            resposta += "\n## 🎯 Insights da Variável Target\n"
            for insight in resumo["target_insights"]:
                resposta += f"- {insight}\n"
        
        return resposta
        
    except Exception as e:
        return f"Erro na análise EDA: {str(e)}"


# Para teste
if __name__ == "__main__":
    # Teste com dados sintéticos
    import numpy as np
    
    np.random.seed(42)
    df_teste = pd.DataFrame({
        'feature_1': np.random.normal(100, 15, 1000),
        'feature_2': np.random.exponential(2, 1000),
        'categoria': np.random.choice(['A', 'B', 'C'], 1000),
        'target': np.random.binomial(1, 0.3, 1000)
    })
    
    # Adicionar alguns nulos e outliers
    df_teste.loc[np.random.choice(df_teste.index, 50), 'feature_1'] = np.nan
    df_teste.loc[np.random.choice(df_teste.index, 20), 'feature_2'] = 50  # outliers
    
    resultado = executar_eda_completa(df_teste, target_column='target')
    print(resultado)