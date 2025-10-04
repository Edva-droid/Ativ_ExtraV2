from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict, List, Any, Dict, NotRequired
from src.agent.agentes_especializados import agente_geral, agente_graficos, agente_conclusoes
from src.util.utils import analise_dados
from src.ml.ml_agent_node import agente_ml_executor, treinar_e_avaliar_modelos_ml, create_tool_calling_agent


from langchain.tools import Tool
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
import re


class AgentState(TypedDict):
    input: str
    output: NotRequired[str]
    chat_history: List[Any]
    intermediate_results: NotRequired[Dict[str, Any]]
    eda_context: NotRequired[Dict[str, Any]]  # Contexto específico para EDA


def clean_numpy(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    return obj

def create_eda_context(df: pd.DataFrame) -> Dict[str, Any]:
    """Cria contexto rico para EDA baseado no dataset"""
    try:
        context = {} # Inicializa o contexto VAZIO
        
        # Cria a seção dataset_info primeiro, protegida contra falhas
        context["dataset_info"] = {
            "shape": (int(df.shape[0]), int(df.shape[1])),
            "columns": df.columns.tolist(),
            "dtypes": df.dtypes.astype(str).to_dict()
        }

        context = {
            "dataset_info": {
                # "shape": df.shape,
                "shape": (int(df.shape[0]), int(df.shape[1])),
                "columns": df.columns.tolist(),
                "dtypes": df.dtypes.astype(str).to_dict()
            },
            "data_quality": {
                # "missing_values": df.isnull().sum().to_dict(),
                "missing_values": {col: int(count) for col, count in df.isnull().sum().items()},
                "duplicates": int(df.duplicated().sum()),
                "memory_usage_mb": round(df.memory_usage(deep=True).sum() / 1024**2, 2)
            },
            "column_types": {
                "numeric": df.select_dtypes(include=['number']).columns.tolist(),
                "categorical": df.select_dtypes(include=['object', 'category']).columns.tolist(),
                "datetime": df.select_dtypes(include=['datetime']).columns.tolist()
            },
            "column_stats": {}  # Nova seção para valores únicos e top valores
        }
        
        # Adicionar estatísticas básicas para colunas numéricas
        numeric_cols = context["column_types"]["numeric"]
        if numeric_cols:
            numeric_stats = df[numeric_cols].describe()
            numeric_summary_py = {
                stat: {
                    # CORREÇÃO CRÍTICA AQUI: Incluir np.number na verificação de tipo
                    k: float(v) if isinstance(v, (int, float, np.number)) else v 
                    for k, v in values.items()
                }
                for stat, values in numeric_stats.to_dict().items()
            }
            context["numeric_summary"] = numeric_summary_py

        # Adicionar valores únicos e top values por coluna
        for col in df.columns:
            col_data = df[col]
            unique_count = int(col_data.nunique())
            top_values = col_data.value_counts().head(5).to_dict()  # top 5 valores
            context["column_stats"][col] = {
                "unique_values": unique_count,
                "top_values": top_values
            }

        # Identificar possível variável target
        potential_targets = []
        target_keywords = ['class', 'target', 'label', 'y', 'outcome', 'result', 'fraud']
        
        for col in df.columns:
            if any(keyword in col.lower() for keyword in target_keywords):
                potential_targets.append(col)
        
        if potential_targets:
            context["potential_targets"] = potential_targets
        print(f"DEBUG Create Eda Context: {context}")
        return json.loads(json.dumps(context, default=clean_numpy))
        
    except Exception as e:
        return {"error": f"Erro ao criar contexto EDA: {str(e)}"}

def enhanced_query_mapper(user_input: str, df: pd.DataFrame, eda_context: Dict[str, Any]) -> str:
    """
    Mapper inteligente para EDA que aplica truncamento seletivo de contexto 
    para garantir que o prompt se encaixe no limite de 6000 tokens do Groq.
    """
    user_input_lower = user_input.lower()
    
     # Para queries simples, NÃO enviar contexto
    if "tipos" in user_input_lower and "dados" in user_input_lower:
        return "df.dtypes.to_dict()" 
    
    elif "outlier" in user_input_lower or "atípico" in user_input_lower:
        # Chamar função direta do eda_direto.py
        from src.agent.eda_direto import analisar_outliers
        return analisar_outliers(df)  # ← Execução direta
    
    # === TRATAMENTO PARA CONSULTAS DE ESTRUTURA (LOW-TOKEN) ===
    # 💡 Se a pergunta for sobre tipos/estrutura, injetamos APENAS o resumo.
    elif any(term in user_input_lower for term in ["tipos de dados", "tipo de dado", "estrutura", "colunas", "quais são os tipos"]):
        
        info = eda_context['dataset_info']
        types = eda_context['column_types']
        
        # Este prompt é pequeno o suficiente (LOW-TOKEN)
        return f"""
                # Contexto do Dataset (Apenas Estrutura)
                Você está analisando um DataFrame para responder à pergunta do usuário, que é sobre a estrutura e tipos de dados.

                ## ESTRUTURA E TIPOS
                - **Shape:** {info['shape'][0]} linhas x {info['shape'][1]} colunas
                - **Tipos de Colunas:**
                    - Numéricas ({len(types['numeric'])}): {', '.join(types['numeric'])}
                    - Categóricas ({len(types['categorical'])}): {', '.join(types['categorical'])}
                    
                # PERGUNTA DO USUÁRIO
                Com base APENAS no resumo de estrutura acima, responda à pergunta: '{user_input}'
                """

    elif any(term in user_input_lower for term in ["outlier", "outliers", "atípico", "atípicos", "anomalia"]):
        return """
            import matplotlib.pyplot as plt
            import numpy as np

            numeric_cols = df.select_dtypes(include=['number']).columns
            print("=== ANÁLISE DE OUTLIERS (IQR) ===\\n")

            for col in numeric_cols[:5]:  # Limitar a 5 colunas
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR
                
                outliers = df[(df[col] < lower) | (df[col] > upper)]
                pct = len(outliers) / len(df) * 100
                
                print(f"{col}: {len(outliers)} outliers ({pct:.2f}%)")

            # Boxplots
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            axes = axes.ravel()

            for i, col in enumerate(numeric_cols[:4]):
                df[col].plot(kind='box', ax=axes[i])
                axes[i].set_title(col)

            plt.tight_layout()
            plt.show()

            "Análise de outliers concluída"
            """

    # === TRATAMENTO PARA OUTRAS CONSULTAS (FULL/TRUNCATED CONTEXT) ===
    # Para qualquer outra consulta, o agente precisa de estatísticas.
    else:
        # 1. TRUNCAMENTO DE SEGURANÇA para limitar o tamanho da requisição (ex: 5 colunas).
        context_copy = eda_context.copy()
        
        # Truncamento de 'column_stats'
        if 'column_stats' in context_copy:
            keys_to_keep = list(context_copy['column_stats'].keys())[:5]
            context_copy['column_stats'] = {k: context_copy['column_stats'][k] for k in keys_to_keep}
        
        # Truncamento de 'numeric_summary'
        if 'numeric_summary' in context_copy:
            keys_to_keep = list(context_copy['numeric_summary'].keys())[:5]
            context_copy['numeric_summary'] = {k: context_copy['numeric_summary'][k] for k in keys_to_keep}
            
        context_str = json.dumps(context_copy, indent=2)

        # 2. Retorna o prompt com o contexto truncado (SEGURANÇA CONTRA ERRO 413)
        print(f"DEBUG Enhanced query mapper: {context_str}")
        return f"""
                # Contexto do Dataset (Detalhes Truncados)
                Use as estatísticas detalhadas abaixo para responder à pergunta. Atenção: O contexto foi truncado para detalhes das primeiras 5 colunas para evitar o estouro de limite de tokens (TPM: 6000). Se a resposta precisar de dados de outras colunas, solicite-os explicitamente.

                ```json
                {context_str}

                # def enhanced_query_mapper(user_input: str, df: pd.DataFrame, eda_context: Dict[str, Any]) -> str:
                #     """


def criar_orquestrador(llm, df: pd.DataFrame):
    """
    Cria o orquestrador LangGraph otimizado para EDA genérica
    """

    analise_dados_tool = Tool(
        name="analise_dados",
        func=lambda pergunta: analise_dados(df, pergunta),
        description="Tool que responde perguntas matemáticas exatas sobre o DataFrame."
    )
    # Criar contexto rico de EDA
    eda_context = create_eda_context(df)
    print(f"DEBUG Criar Orquestrador: {eda_context}")
    # Inicializar os agentes com o contexto
    agente_geral_executor = agente_geral(llm, df, extra_tools=[analise_dados_tool])

    agente_graficos_executor = agente_graficos(llm, df)
    
    # Tentar criar agente ML se disponível
    try:
        from src.agent.agentes_especializados import agente_machine_learning
        agente_ml_executor = agente_machine_learning(llm, df)
    except:
        agente_ml_executor = None
    
    agente_conclusoes_executor = agente_conclusoes(
        llm, agente_geral_executor, agente_graficos_executor, agente_ml_executor
    )

    def classificar_intencao_llm(user_input: str, llm) -> str:
        """
        Usa o LLM para classificar a intenção do usuário
        Retorna: 'analise_geral', 'grafico', 'conclusao' ou 'fallback'
        """
        prompt = f"""
    Você é um classificador de intenção de usuário para análise de dados. Sua única função é retornar a categoria correta para a query.
 
    REGRAS:
    1. A resposta deve ser **uma única palavra**, **minúscula** e **sem pontuação** (ex: conclusao, grafico, ml).
    2. Use 'conclusao' para perguntas que buscam resumo, síntese, insights finais ou o significado dos resultados.
    3. Use 'analise_geral' para estatísticas, cálculos, valores específicos, contagem ou análise descritiva.

    CATEGORIAS VÁLIDAS:
    analise_geral, grafico, ml, conclusao, fallback

    Query para classificar: "{user_input}"

    RESPOSTA (APENAS A CATEGORIA):
    """
        # result = llm.invoke({"input": prompt})
        result = llm.invoke(prompt)
        intent = (result.content or "fallback").strip().lower()
        if intent not in ["analise_geral", "grafico", "ml", "conclusao"]:
            intent = "fallback"
        print(f"DEBUG Classificar intenção: {intent}")
        return intent
    

    def decidir_proximo(state: dict) -> str:
        """
        Decide o próximo nó baseado na intenção do usuário usando LLM.
        Esta função é chamada APÓS o nó "analise_de_dados_geral" ser executado.
        """
        intent = classificar_intencao_llm(state.get("input", ""), llm)
        print(f"DEBUG Decidir Proximo: {intent}")
        if intent == "grafico":
            return "gerador_de_graficos"
        
        elif intent == "conclusao":
            # A análise EDA/Geral já ocorreu (pois você acabou de sair dela), 
            # então vá direto para as conclusões.
            return "sintetizador_de_conclusoes"
            
        elif intent == "analise_geral":
            # Se era só uma análise geral e já foi feita, finalize.
            return "analise_geral"  # Leva ao END
        
        elif intent == "ml":  # 🔑 nova rota para ML
            return "executor_ml"
    
        else:  # fallback
            return "fallback"
        

    def node_agente_geral(state: AgentState) -> Dict[str, Any]:
        """Agente Geral"""
        try:
            query = state["input"].lower()
            
            # 1. TENTAR FUNÇÕES DIRETAS PRIMEIRO
            from src.agent.eda_direto import FUNCOES_EDA
            for keyword, funcao in FUNCOES_EDA.items():
                if keyword in query:
                    output = funcao(df)
                    return {
                        "output": output,
                        "intermediate_results": {"analise_eda": output}
                    }
            
            # 2. SE NÃO ENCONTROU, USAR CÓDIGO MAPEADO
            codigo_mapeado = enhanced_query_mapper(state["input"], df, eda_context)
            
            # Se o mapper retornou código Python simples, executar diretamente
            if len(codigo_mapeado) < 200 and not "```" in codigo_mapeado:
                try:
                    result = eval(codigo_mapeado, {}, {"df": df, "pd": pd})
                    return {"output": str(result)}
                except:
                    pass  # Se falhar, vai para o agente
            
            # 3. ÚLTIMO RECURSO: Agente LLM
            result = agente_geral_executor.invoke({"input": codigo_mapeado})
            # Execução com query aprimorada
            output = result.get("output", "Sem resposta da análise EDA")
            
            # Se o LLM não devolveu nada no campo "output", tentar pegar do último passo
            if not output and result.get("intermediate_steps"):
                last_step = result["intermediate_steps"][-1]
                if isinstance(last_step, str):
                    output = last_step
                    
            # Se o output for código Python com print(), executar e capturar
            if isinstance(output, str) and "print(" in output:
                import io, contextlib
                buffer = io.StringIO()
                local_vars = {"df": df}
                with contextlib.redirect_stdout(buffer):
                    exec(output, {}, local_vars)
                output = buffer.getvalue()
            
            # Garantir que o output seja string
            if isinstance(output, (dict, list)):
                output = json.dumps(output, ensure_ascii=False, indent=2)
            elif not isinstance(output, str):
                output = str(output)
                
            return {
                "output": output,
                "intermediate_results": {"analise_eda": output},
                "eda_context": eda_context
            }
        except Exception as e:
            error_msg = f"Erro na análise EDA: {str(e)}"
            return {
                "output": error_msg,
                "intermediate_results": {"analise_eda": error_msg},
                # usa o contexto limpo se deu tempo de gerar, caso contrário um dict vazio
                "eda_context": eda_context or {}
            }


        
    MAX_ROWS_FOR_PLOTS = 5000  # Limite para gerar gráficos
    def node_agente_graficos(state: AgentState) -> Dict[str, Any]:
        """Nó para geração de gráficos sob demanda"""
        print("🎨 DEBUG node_agente_graficos | Chamado para gerar gráficos")
        try:
            # Subamostra para performance
            if df.shape[0] > MAX_ROWS_FOR_PLOTS:
                df_plot = df.sample(n=MAX_ROWS_FOR_PLOTS, random_state=42)
            else:
                df_plot = df

            # Chamada ao executor de gráficos
            result = agente_graficos_executor.invoke({"input": state["input"], "df": df_plot})
            output = result.get("output", None)
            
            # Se o executor gerou uma figura Matplotlib
            if isinstance(output, plt.Figure):
                st.pyplot(output)   # 🔑 exibe diretamente no Streamlit
                output_display = f"[Gráfico exibido: {state['input']}]"
            else:
                output_display = str(output)

            # Guardar no estado
            intermediate = state.get("intermediate_results", {})
            intermediate["graficos"] = output_display

            return {
                "output": output_display,
                "intermediate_results": intermediate
            }

        except Exception as e:
            error_msg = f"Erro no agente de gráficos: {str(e)}"
            intermediate = state.get("intermediate_results", {})
            intermediate["graficos"] = error_msg
            return {
                "output": error_msg,
                "intermediate_results": intermediate
            }

    def node_agente_conclusoes(state: AgentState) -> Dict[str, Any]:
        """Nó para síntese de conclusões"""
        try:
            # Preparar contexto com resultados anteriores
            context = f"Pergunta do usuário: {state['input']}\n"
            if state.get("intermediate_results"):
                for key, value in state["intermediate_results"].items():
                    context += f"\nResultado {key}: {value}\n"
            
            result = agente_conclusoes_executor.invoke({
                "input": context,
                "chat_history": state.get("chat_history", [])
            })
            
            output = result.get("output", "Não foi possível gerar conclusões")
            
            if isinstance(output, (dict, list)):
                output = json.dumps(output, ensure_ascii=False, indent=2)
            elif not isinstance(output, str):
                output = str(output)
                
            intermediate = state.get("intermediate_results", {})
            intermediate["conclusoes"] = output
            print(f"DEBUG Node Agente Conclusões OUTPUT: {output}")
            print()
            print(f"DEBUG Node Agente Conclusões Intermediate: {intermediate}")

            return {
                "output": output,
                "intermediate_results": intermediate
            }
        except Exception as e:
            error_msg = f"Erro no agente de conclusões: {str(e)}"
            intermediate = state.get("intermediate_results", {})
            intermediate["conclusoes"] = error_msg
            
            return {
                "output": error_msg,
                "intermediate_results": intermediate
            }


    # 💡 NOVO NÓ: Chamada direta à tool analise_dados
    def node_analise_rapida(state: AgentState) -> Dict[str, Any]:
        """Nó para executar a tool analise_dados diretamente (cálculos rápidos)"""
        pergunta = state["input"]
        
        # Chamada direta à função que está na tool
        output_tool = analise_dados(df, pergunta)
        
        # Se a tool respondeu com o resultado, o trabalho acabou.
        if not output_tool.startswith("❓") and not output_tool.startswith("⚠️"):
            # Se a tool conseguiu responder (sem erro ou fallback)
            return {
                "output": output_tool,
                "intermediate_results": {"analise_rapida": output_tool},
                # Não mexer no eda_context aqui
            }
        else:
            # Se a tool não soube responder, encaminha para o agente geral
            # Mantém a entrada original
            return {
                "output": None, # Não gera output final ainda
                "intermediate_results": {"analise_rapida": output_tool}, # Guarda o feedback da tool
                "next_step": "analise_de_dados_geral" # Sinaliza que deve ir para o próximo nó
            }


    def node_executor_ml(state: AgentState) -> Dict[str, Any]:
        """Nó para execução de análises de Machine Learning"""
        try:
            result = agente_ml_executor.invoke({"input": state["input"]})
            output = result.get("output", "Sem resultado do agente de ML")

            if isinstance(output, (dict, list)):
                output = json.dumps(output, ensure_ascii=False, indent=2)
            elif not isinstance(output, str):
                output = str(output)

            intermediate = state.get("intermediate_results", {})
            return {
                "output": None,  # resposta final será sintetizada
                "intermediate_results": {**intermediate, "ml_resultados": output}
            }
        except Exception as e:
            error_msg = f"Erro no agente de ML: {str(e)}"
            intermediate = state.get("intermediate_results", {})
            return {
                "output": None,
                "intermediate_results": {**intermediate, "ml_resultados": error_msg}
            }
    

    # Criar o workflow
    workflow = StateGraph(AgentState)
    
    # Adicionar nós
    workflow.add_node("analise_rapida", node_analise_rapida) # <--- Adicionado
    workflow.add_node("analise_de_dados_geral", node_agente_geral)
    workflow.add_node("gerador_de_graficos", node_agente_graficos)
    workflow.add_node("sintetizador_de_conclusoes", node_agente_conclusoes)
    workflow.add_node("executor_ml", node_executor_ml)
    
    # Definir ponto de entrada
    # workflow.set_entry_point("analise_de_dados_geral")
    workflow.set_entry_point('analise_rapida')
    # 💡 NOVO ROTEAMENTO: Decide o que fazer após a tool de cálculo (analise_rapida)
    workflow.add_conditional_edges(
        "analise_rapida",
        # A função lambda verifica se a tool pediu o próximo passo ('analise_de_dados_geral')
        # Se não houver 'next_step' no estado, significa que a tool respondeu e retorna END.
        lambda state: state.get("next_step", "END"), 
        {
            "analise_de_dados_geral": "analise_de_dados_geral", # Se next_step for 'analise_de_dados_geral', vai para o LLM
            "END": END, # Se a tool respondeu, o LangGraph para aqui.
        }
    )

    # Roteamento original (após análise geral, decide se vai para gráfico, conclusão ou fim)
    
    workflow.add_conditional_edges(
        "analise_de_dados_geral",
        decidir_proximo,
        {
            "gerador_de_graficos": "gerador_de_graficos",     # Chave 1
            "sintetizador_de_conclusoes": "sintetizador_de_conclusoes",  # Chave 2
            "analise_geral": END,                             # Chave 3
            "executor_ml": "executor_ml",                     # Chave 4
            "fallback": END                                   # Chave 5
        }
    )
 
    # Gráficos sempre levam a conclusões
    workflow.add_edge("gerador_de_graficos", "sintetizador_de_conclusoes")
    workflow.add_edge("executor_ml", "sintetizador_de_conclusoes")
    workflow.add_edge("sintetizador_de_conclusoes", END)
    
    # Compilar com checkpointer
    checkpointer = MemorySaver()
    return workflow.compile(checkpointer=checkpointer)


# Teste local
if __name__ == "__main__":
    from langchain_groq import ChatGroq
    from src.agent.config_agent import Settings
    from src.data.bases import download_base
    
    # Carregar dados
    df = download_base()
    llm = ChatGroq(
        temperature=0,
        # groq_api_key=Settings.GROQ_API_KEY,
        model=Settings.LLM_MODEL
    )
    
    # Criar orquestrador
    graph = criar_orquestrador(llm, df)
    
    # 💡 1. DEFINIR CONFIGURAÇÃO COM UM ID ÚNICO (thread_id)
    # Este ID permite que o checkpointer MemorySaver salve o estado.
    config = {"configurable": {"thread_id": "minha_sessao_eda_teste"}}

    # Teste
    result = graph.invoke({
        "input": "Existem agrupamentos (clusters) nos dados?",
        "chat_history": []
    },
    config= config
    )
    
    print("Resultado final:", result)