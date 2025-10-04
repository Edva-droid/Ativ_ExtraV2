import pandas as pd
import matplotlib.pyplot as plt
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.tools import Tool
from typing import Dict, Any
import json

# Import das ferramentas de ML
from src.ml.ml_tools import (
    analyze_data_for_ml, 
    train_ml_models, 
    create_ml_comparison_plots,
    DataPreprocessor,
    ModelTrainer,
    ModelEvaluator,
    ML_AVAILABLE
)


def create_ml_analysis_tool(df: pd.DataFrame):
    """Cria ferramenta de análise de dados para ML"""
    def ml_data_analysis(query: str) -> str:
        """Analisa dados para preparação de Machine Learning"""
        try:
            result = analyze_data_for_ml(df)
            print(f"DEBUG ML data Analisis Tool: {result}")
            return result
        except Exception as e:
            return f"Erro na análise ML: {str(e)}"
    
    return Tool(
        name="analise_dados_ml",
        func=ml_data_analysis,
        description="Analisa dados para Machine Learning - qualidade, tipos, missing values, distribuições"
    )


def create_model_training_tool(df: pd.DataFrame):
    """Cria ferramenta de treinamento de modelos"""
    def train_models(query: str) -> str:
        """Treina múltiplos modelos de ML"""
        try:
            # Extrair coluna target da query (implementação simples)
            words = query.lower().split()
            
            # Procurar por indicações da coluna target
            target_column = None
            possible_targets = []
            
            # Palavras-chave que indicam target
            target_keywords = ["target", "classe", "class", "label", "objetivo", "prever", "predizer"]
            
            # Procurar colunas mencionadas na query
            for col in df.columns:
                if col.lower() in query.lower():
                    possible_targets.append(col)
            
            # Procurar por padrões comuns
            for word in words:
                if word in target_keywords:
                    idx = words.index(word)
                    if idx + 1 < len(words):
                        potential_target = words[idx + 1]
                        for col in df.columns:
                            if col.lower() == potential_target or potential_target in col.lower():
                                target_column = col
                                break
            
            # Se não encontrou, usar a primeira coluna candidata
            if not target_column and possible_targets:
                target_column = possible_targets[0]
            
            # Fallback: usar a última coluna (convenção comum)
            if not target_column:
                target_column = df.columns[-1]
            result = train_ml_models(df, target_column)
            print(f"DEBUG Train Models : {result}")
            return result
            
        except Exception as e:
            return f"Erro no treinamento: {str(e)}"
    
    return Tool(
        name="treinar_modelos_ml",
        func=train_models,
        description="Treina múltiplos modelos de Machine Learning e compara performance"
    )


def create_ml_visualization_tool(df: pd.DataFrame):
    """Cria ferramenta de visualização ML"""
    def create_ml_plots(query: str) -> str:
        """Cria gráficos de comparação de modelos"""
        try:
            # Lógica similar para extrair target
            target_column = df.columns[-1]  # Simplificado para exemplo
            
            words = query.lower().split()
            for col in df.columns:
                if col.lower() in query.lower():
                    target_column = col
                    break
            
            fig = create_ml_comparison_plots(df, target_column)
            return "Gráfico de comparação de modelos ML criado com sucesso"
            
        except Exception as e:
            return f"Erro na criação de gráficos ML: {str(e)}"
    
    return Tool(
        name="graficos_comparacao_ml",
        func=create_ml_plots,
        description="Cria gráficos de comparação de performance entre modelos de ML"
    )


def create_ml_recommendations_tool(df: pd.DataFrame):
    """Cria ferramenta de recomendações ML"""
    def ml_recommendations(query: str) -> str:
        """Fornece recomendações baseadas na análise dos dados"""
        try:
            preprocessor = DataPreprocessor()
            analysis = preprocessor.analyze_data_quality(df)
            
            recommendations = []
            print(f"DEBUG ML Recommendations: {analysis}")
            # Análise de qualidade dos dados
            missing_total = sum(analysis["missing_values"].values())
            if missing_total > 0:
                recommendations.append(f"⚠️ Encontrados {missing_total} valores faltantes. Considere estratégias de imputação.")
            
            # Análise de balanceamento (para colunas categóricas)
            if analysis["categorical_columns"]:
                for col in analysis["categorical_columns"]:
                    unique_count = df[col].nunique()
                    if unique_count == 2:
                        value_counts = df[col].value_counts()
                        ratio = value_counts.min() / value_counts.max()
                        if ratio < 0.3:
                            recommendations.append(f"⚖️ Coluna '{col}' está desbalanceada (ratio: {ratio:.2f}). Considere técnicas de balanceamento.")
            
            # Recomendações de modelos baseado no tipo de dados
            numeric_cols = len(analysis["numeric_columns"])
            categorical_cols = len(analysis["categorical_columns"])
            
            if numeric_cols > categorical_cols:
                recommendations.append("📊 Dataset com predominância numérica. Modelos recomendados: Random Forest, XGBoost, SVM.")
            else:
                recommendations.append("🏷️ Dataset com muitas variáveis categóricas. Considere encoding adequado e modelos como Random Forest.")
            
            # Análise de tamanho
            n_samples = analysis["shape"][0]
            if n_samples < 1000:
                recommendations.append("📏 Dataset pequeno (<1000 amostras). Considere cross-validation e modelos menos complexos.")
            elif n_samples > 100000:
                recommendations.append("📈 Dataset grande (>100k amostras). Considere modelos escaláveis como XGBoost ou técnicas de sampling.")
            
            # Análise de dimensionalidade
            n_features = analysis["shape"][1]
            if n_features > n_samples:
                recommendations.append("🔍 Mais features que amostras. Considere redução de dimensionalidade (PCA) ou regularização.")
            
            # Análise de memória
            if analysis["memory_usage_mb"] > 1000:
                recommendations.append("💾 Dataset usa muita memória (>1GB). Considere otimização de tipos de dados ou processamento em chunks.")
            
            result = {
                "data_summary": {
                    "samples": n_samples,
                    "features": n_features,
                    "numeric_features": numeric_cols,
                    "categorical_features": categorical_cols,
                    "missing_values": missing_total,
                    "memory_usage_mb": analysis["memory_usage_mb"]
                },
                "recommendations": recommendations,
                "suggested_workflow": [
                    "1. 🔍 Análise exploratória completa",
                    "2. 🧹 Limpeza e tratamento de missing values",
                    "3. 🔄 Encoding de variáveis categóricas",
                    "4. ⚖️ Verificar balanceamento das classes",
                    "5. 📏 Normalização/padronização dos dados",
                    "6. 🤖 Treinamento de múltiplos modelos",
                    "7. 📊 Avaliação e comparação de performance",
                    "8. 🎯 Otimização de hiperparâmetros do melhor modelo"
                ]
            }
            print(f"DEBUG: {json.dumps(result, ensure_ascii=False, indent=2)}")
            return json.dumps(result, ensure_ascii=False, indent=2)
            
        except Exception as e:
            return f"Erro nas recomendações ML: {str(e)}"
    
    return Tool(
        name="recomendacoes_ml",
        func=ml_recommendations,
        description="Fornece recomendações e insights para projetos de Machine Learning baseado nos dados"
    )


def agente_machine_learning(llm: BaseChatModel, df: pd.DataFrame) -> AgentExecutor:
    """
    Agente especialista em Machine Learning
    """
    # Verificar disponibilidade das bibliotecas ML
    if not ML_AVAILABLE:
        # Criar ferramenta de aviso
        def ml_warning(query: str) -> str:
            return """
            ⚠️ BIBLIOTECAS DE MACHINE LEARNING NÃO DISPONÍVEIS
            
            Para usar as funcionalidades de ML, instale as dependências:
            
            pip install scikit-learn xgboost imbalanced-learn
            
            Funcionalidades disponíveis após instalação:
            - Análise de qualidade dos dados
            - Treinamento automático de múltiplos modelos
            - Comparação de performance
            - Otimização de hiperparâmetros
            - Visualizações de comparação
            - Recomendações personalizadas
            """
        
        tools = [Tool(
            name="aviso_ml",
            func=ml_warning,
            description="Informa sobre a necessidade de instalar bibliotecas ML"
        )]
    else:
        # Criar todas as ferramentas ML
        tools = [
            create_ml_analysis_tool(df),
            create_model_training_tool(df),
            create_ml_visualization_tool(df),
            create_ml_recommendations_tool(df)
        ]
    
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            """Você é um especialista em Machine Learning e Data Science.

SUAS CAPACIDADES:
- Análise de qualidade de dados para ML
- Treinamento automático de múltiplos modelos
- Comparação de performance entre modelos
- Recomendações baseadas nas características dos dados
- Criação de visualizações de comparação

TIPOS DE ANÁLISE QUE VOCÊ PODE FAZER:
1. 🔍 ANÁLISE DE DADOS: Qualidade, tipos, missing values, distribuições
2. 🤖 TREINAMENTO: Múltiplos modelos (Random Forest, XGBoost, SVM, etc.)
3. 📊 COMPARAÇÃO: Métricas de performance, gráficos, rankings
4. 🎯 RECOMENDAÇÕES: Sugestões baseadas nos dados

WORKFLOW RECOMENDADO:
1. Sempre comece com análise de dados
2. Identifique o tipo de problema (classificação/regressão)
3. Treine múltiplos modelos
4. Compare performances
5. Forneça recomendações

IMPORTANTE:
- Sempre analise os dados antes de treinar modelos
- Explique os resultados de forma clara
- Forneça insights acionáveis
- Considere limitações e sugestões de melhoria

Responda de forma didática e orientada a resultados.
"""
        ),
        HumanMessagePromptTemplate.from_template("{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])

    agente = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(
        agent=agente, 
        tools=tools, 
        verbose=True,
        handle_parsing_errors=True,
        return_intermediate_steps=True
    )


def agente_ml_avancado(llm: BaseChatModel, df: pd.DataFrame) -> AgentExecutor:
    """
    Versão avançada do agente ML com funcionalidades extras
    """
    def advanced_ml_pipeline(query: str) -> str:
        """Pipeline completo de ML"""
        try:
            if not ML_AVAILABLE:
                return "Bibliotecas ML não disponíveis. Instale: pip install scikit-learn xgboost imbalanced-learn"
            
            # Detectar coluna target automaticamente
            target_col = None
            
            # Estratégias para detectar target:
            # 1. Procurar palavras-chave na query
            query_lower = query.lower()
            for col in df.columns:
                if col.lower() in query_lower:
                    target_col = col
                    break
            
            # 2. Procurar colunas com nomes típicos de target
            if not target_col:
                target_names = ['target', 'class', 'label', 'y', 'outcome', 'result']
                for col in df.columns:
                    if col.lower() in target_names:
                        target_col = col
                        break
            
            # 3. Usar a última coluna (convenção)
            if not target_col:
                target_col = df.columns[-1]
            
            # Executar pipeline completo
            results = {
                "1_data_analysis": json.loads(analyze_data_for_ml(df)),
                "2_model_training": json.loads(train_ml_models(df, target_col)),
                "3_recommendations": "Pipeline completo executado com sucesso"
            }
            print(f"DEBUG Advanced ml Pipeline: {json.dumps(results, ensure_ascii=False, indent=2)}")
            return json.dumps(results, ensure_ascii=False, indent=2)
            
        except Exception as e:
            return f"Erro no pipeline ML: {str(e)}"
    
    # Ferramentas básicas + pipeline avançado
    tools = []
    
    if ML_AVAILABLE:
        tools.extend([
            create_ml_analysis_tool(df),
            create_model_training_tool(df),
            create_ml_visualization_tool(df),
            create_ml_recommendations_tool(df),
            Tool(
                name="pipeline_ml_completo",
                func=advanced_ml_pipeline,
                description="Executa pipeline completo de ML: análise + treinamento + recomendações"
            )
        ])
    else:
        tools.append(Tool(
            name="aviso_ml",
            func=lambda x: "Bibliotecas ML não disponíveis. Instale as dependências necessárias.",
            description="Aviso sobre bibliotecas ML"
        ))
    
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            """Você é um Data Scientist sênior especialista em Machine Learning.

MISSÃO: Automatizar completamente projetos de ML, desde análise até recomendações.

FERRAMENTAS DISPONÍVEIS:
🔍 analise_dados_ml: Análise detalhada da qualidade dos dados
🤖 treinar_modelos_ml: Treinamento automático de múltiplos modelos
📊 graficos_comparacao_ml: Visualizações de performance
🎯 recomendacoes_ml: Insights e sugestões personalizadas
⚡ pipeline_ml_completo: Pipeline end-to-end automatizado

ABORDAGEM:
1. Para análises rápidas: use pipeline_ml_completo
2. Para análises detalhadas: use ferramentas individuais
3. Sempre forneça contexto e interpretação dos resultados
4. Sugira próximos passos e melhorias

FORMATO DE RESPOSTA:
- Seja claro e didático
- Use emojis para organizar informações
- Forneça insights acionáveis
- Explique limitações quando relevante

Você é o especialista que transforma dados em insights valiosos!
"""
        ),
        HumanMessagePromptTemplate.from_template("{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])

    agente = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(
        agent=agente, 
        tools=tools, 
        verbose=True,
        handle_parsing_errors=True,
        return_intermediate_steps=True
    )


# Teste local
if __name__ == "__main__":
    from langchain_groq import ChatGroq
    from src.agent.config_agent import Settings
    from src.data.bases import download_base
    
    # Carregar dados
    df = download_base()
    print("DataFrame carregado:", df.shape)
    
    # Inicializar LLM
    llm = ChatGroq(
        temperature=0, 
        # groq_api_key=Settings.GROQ_API_KEY, 
        model=Settings.LLM_MODEL
    )

    # Teste do agente ML
    executor = agente_machine_learning(llm, df)
    resultado = executor.invoke({"input": "Analise os dados e treine modelos de machine learning"})
    print("Resultado:", resultado.get("output"))