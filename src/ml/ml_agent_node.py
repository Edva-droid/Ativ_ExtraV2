from pydantic import BaseModel, Field # ✅ Agora direto do pydantic v2
from langchain_core.tools import tool  # ✅ Tools modernas
from langchain.agents import create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
import pandas as pd
# Importa sua função real de ML
from src.ml.ml_tools import train_ml_models  


# ----------------------------------------------------
# 1. Definição da Tool (Interface para o LLM)
# ----------------------------------------------------
class MLModelInput(BaseModel):
    """Input para a ferramenta de treinamento e avaliação de modelos de Machine Learning."""
    target_column: str = Field(
        description="O nome exato da coluna alvo (target) que o modelo deve prever. É obrigatório."
    )
    test_size: float = Field(
        default=0.2,
        description="A proporção do dataset a ser usada para teste (ex: 0.2 para 20%)."
    )


@tool(args_schema=MLModelInput)
def treinar_e_avaliar_modelos_ml(target_column: str, test_size: float = 0.2) -> str:
    """
    Treina e avalia automaticamente múltiplos modelos de Machine Learning 
    (Classificação ou Regressão) para uma coluna alvo ('target_column').
    Retorna um resumo JSON dos resultados e métricas.
    """
    # ⚠️ Pega o DataFrame do st.session_state ou de outro state global
    import streamlit as st
    if "current_df" not in st.session_state or st.session_state.current_df.empty:
        return "❌ Nenhum DataFrame carregado para análise de ML."

    df = st.session_state.current_df
    return train_ml_models(df, target_column, test_size)


# Lista de ferramentas disponíveis
ml_tools = [treinar_e_avaliar_modelos_ml]


# ----------------------------------------------------
# 2. Definição do Nó (Executor do Agente)
# ----------------------------------------------------

def agente_ml_executor(state: dict) -> dict:
    """
    Executor do nó de Machine Learning.
    Recebe o estado atual do LangGraph e retorna um dicionário com o output.
    """
    try:
        # Recupera instrução do usuário
        user_input = state.get("input", "")
        
        # Aqui você poderia montar um prompt específico
        ml_prompt = ChatPromptTemplate.from_messages([
            ("system", "Você é um assistente de Machine Learning."),
            ("user", "{input}")
        ])

        # ⚠️ Precisa de uma instância de LLM (ex: ChatGroq)
        from src.agent.config_agent import Settings
        from langchain_groq import ChatGroq
        llm_ml = ChatGroq(
            temperature=0,
            # groq_api_key=Settings.GROQ_API_KEY,
            model=Settings.LLM_MODEL
        )

        # Cria o agente com a tool
        agente_ml = create_tool_calling_agent(llm_ml, ml_tools, ml_prompt)

        # Executa a chain passando o estado
        result = agente_ml.invoke({"input": user_input})

        return {"output": result.get("output", "⚠️ Nenhum resultado retornado pelo agente ML.")}
    
    except Exception as e:
        return {"output": f"❌ Erro no agente ML: {e}"}
