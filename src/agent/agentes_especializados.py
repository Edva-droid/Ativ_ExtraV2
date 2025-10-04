import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_experimental.tools.python.tool import PythonAstREPLTool
from langchain.tools import Tool, BaseTool
from typing import List
from tqdm import tqdm
from time import time

import inspect
# Import da nova função EDA
try:
    from src.agent.eda_utils import executar_eda_completa, analise_eda_completa
    EDA_MELHORADA_DISPONIVEL = True
except ImportError:
    EDA_MELHORADA_DISPONIVEL = False


def process_output(x):
    """Processa a saída para ser mais legível"""
    if isinstance(x, pd.Series):
        return x.to_dict()
    elif isinstance(x, pd.DataFrame):
        if len(x) > 10:
            return {
                "shape": x.shape,
                "columns": x.columns.tolist(),
                "dtypes": x.dtypes.to_dict(),
                "sample": x.head().to_dict()
            }
        result = x.to_dict()
        print(f"DEBUG Process output: {result}")
        return result
    elif isinstance(x, (list, dict)):
        return x
    else:
        result = str(x)
        print(f"DEBUG: Resultado processado: {str(result)[:200]}") 
        return result


def agente_geral(llm: BaseChatModel, df: pd.DataFrame, extra_tools = None) -> AgentExecutor:
    """
    Agente especialista em análise de DataFrames pandas com EDA melhorada.
    """
    from pydantic import BaseModel, Field
    from langchain.tools import StructuredTool

    tools = []
    
    def create_safe_repl_tool(df):
        class CodeInput(BaseModel):
            code: str = Field(description="Código Python a ser executado")
        
        def safe_python_exec(code: str) -> str:
            clean_code = code.strip()
            try:
                local_env = {"df": df, "pd": pd, "json": json}
                print(f"[DEBUG: Linha {inspect.currentframe().f_lineno}] Executando código safe_python: {clean_code}")
                
                result = eval(clean_code, {}, local_env)
                processed = process_output(result)
                
                print(f"DEBUG: Resultado final: {processed}")
                return processed
                
            except Exception as e:
                error_msg = f"Erro na execução: {str(e)}"
                print(f"DEBUG: {error_msg}")
                return error_msg
        
        return StructuredTool.from_function(
            func=safe_python_exec,
            name="python_repl",
            description="Executa expressões Python no DataFrame 'df'",
            args_schema=CodeInput
        )
    
    tools = [create_safe_repl_tool(df)]
    
    if extra_tools:
        tools.extend(extra_tools)
    
    # Ferramenta EDA com schema correto
    if globals().get("EDA_MELHORADA_DISPONIVEL", True):
        class EDAInput(BaseModel):
            query: str = Field(
                description="Descrição da análise EDA desejada",
                default="Análise exploratória completa"
            )
        
        def eda_completa_tool(query: str = "Análise exploratória completa") -> str:
            """Executa análise EDA completa melhorada"""
            try:
                if not query or query.strip() == '':
                    query = "Resumo executivo e principais conclusões da EDA"
                
                target_col = None
                query_lower = query.lower()
                
                for col in df.columns:
                    if col.lower() in query_lower or any(
                        keyword in col.lower() 
                        for keyword in ['target', 'class', 'label']
                    ):
                        target_col = col
                        break
                
                result = executar_eda_completa(df, target_column=target_col)
                print(f"DEBUG EDA Completa Tool: {result[:200]}...")
                return result
            except Exception as e:
                return f"Erro na análise EDA completa: {str(e)}"
        
        tools.append(StructuredTool.from_function(
            func=eda_completa_tool,
            name="eda_completa",
            description="Executa análise exploratória completa com insights automáticos e gráficos",
            args_schema=EDAInput
        ))

    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            """Você é um assistente especialista em análise de dados com pandas.

FERRAMENTAS DISPONÍVEIS:
1. **python_repl(code)** - Executa código Python no DataFrame 'df'
2. **eda_completa(query)** - Análise exploratória completa

REGRAS:
- Para consultas específicas: use python_repl
- Para análises gerais: use eda_completa
- SEMPRE forneça os argumentos nomeados: python_repl(code="...") ou eda_completa(query="...")

EXEMPLOS CORRETOS:
❌ ERRADO: python_repl("df.shape")
✅ CORRETO: python_repl(code="df.shape")

❌ ERRADO: eda_completa()
✅ CORRETO: eda_completa(query="Análise completa dos dados")

Para "Quais são os tipos de dados?":
→ python_repl(code="df.dtypes.to_dict()")

Para "Análise exploratória completa":
→ eda_completa(query="Análise exploratória completa dos dados")
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


def agente_graficos(llm: BaseChatModel, df: pd.DataFrame) -> AgentExecutor:
    """
    Agente especialista em gerar gráficos com EDA melhorada.
    """
    tools = []
    
    # # Ferramenta padrão para gráficos personalizados

    def create_safe_repl_tool(df):
        def safe_python_exec(code: str) -> str:
            clean_code = code.strip()
            try:
                local_env = {"df": df, "pd": pd, "plt": plt, "sns": sns}
                
                # Para debug: log do código recebido
                print(f"DEBUG: Executando código: {clean_code}")
                
                # Executar código diretamente
                # result = exec(clean_code, {}, local_env)
                exec(clean_code, {}, local_env)
                # processed = process_output(result)
                
                # print(f"DEBUG: Resultado final: {processed}")
                # Como é um gráfico (efeito colateral), retorna sucesso
                return "Comando de plotagem executado com sucesso. O gráfico foi gerado."
                
            except Exception as e:
                error_msg = f"Erro na execução: {str(e)}"
                print(f"DEBUG: {error_msg}")
                return error_msg
        
        return Tool(
            name="python_repl",
            func=safe_python_exec,
            description="Executa expressões Python no DataFrame 'df'"
        )
    tools = [create_safe_repl_tool(df)]
    
    # Ferramenta para gráficos automáticos da EDA melhorada
    if EDA_MELHORADA_DISPONIVEL:
        def graficos_eda_auto(query: str) -> str:
            """Gera gráficos automáticos da EDA completa"""
            try:
                # Identificar target se mencionado
                target_col = None
                for col in df.columns:
                    if col.lower() in query.lower():
                        target_col = col
                        break
                
                # Executar análise que já gera os gráficos
                resultados = analise_eda_completa(df, salvar_graficos=False, target_column=target_col)
                print(f'DEBUG Fragicos eda auto: {resultados}')
                return f"Gráficos automáticos gerados: {', '.join(resultados['resumo_executivo']['graficos_gerados'])}"
            except Exception as e:
                return f"Erro na geração automática de gráficos: {str(e)}"
        
        tools.append(Tool(
            name="graficos_automaticos",
            func=graficos_eda_auto,
            description="Gera automaticamente gráficos de distribuição, correlação e análise exploratória"
        ))

    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            """Você é um especialista em visualização de dados com matplotlib e seaborn.

FERRAMENTAS DISPONÍVEIS:
1. **Python REPL**: Para gráficos personalizados
2. **Gráficos Automáticos**: Para conjunto completo de gráficos EDA

QUANDO USAR:
- **Gráficos Automáticos** para:
  * "Crie gráficos para análise exploratória"
  * "Gere visualizações automáticas"
  * "Mostre gráficos de distribuição"
  * "Visualizações completas"

- **Python REPL** para:
  * Gráficos específicos de uma coluna
  * Personalizações especiais
  * Tipos específicos de gráfico

REGRAS PARA GRÁFICOS MANUAIS:
1. Use plt.figure(figsize=(10, 6)) antes de criar gráficos
2. Sempre adicione títulos e rótulos
3. Use plt.tight_layout() no final
4. Para categóricas: countplot, barplot
5. Para numéricas: histogram, boxplot, scatterplot
6. Não use plt.show(). O código deve terminar salvando o gráfico em um arquivo chamado grafico_analise.png. 
7. Sempre use plt.savefig('grafico_analise.png') como o último comando do bloco de código.  
8. Não escreva strings no final, exibir o gráfico.

EXEMPLO DE CÓDIGO:
```python
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='coluna', bins=30)
plt.title('Distribuição de Coluna')
plt.show()
plt.savefig('grafico_analise.png')
"Gráfico gerado com sucesso"
```
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
        handle_parsing_errors=True
    )


def agente_machine_learning(llm: BaseChatModel, df: pd.DataFrame) -> AgentExecutor:
    """
    Agente especialista em Machine Learning
    """
    tools = []
    
    try:
        # Tentar importar ferramentas ML
        from src.ml.ml_tools import (
            analyze_data_for_ml, 
            train_ml_models, 
            create_ml_comparison_plots,
            ML_AVAILABLE
        )
        
        if ML_AVAILABLE:
            def ml_analysis_tool(query: str) -> str:
                """Análise de dados para ML"""
                result = analyze_data_for_ml(df)
                print(f"DEBUG ML Analysis Tool: {result}")
                return result
            
            def ml_training_tool(query: str) -> str:
                """Treinamento de modelos ML"""
                # Tentar extrair coluna target da query
                target_col = df.columns[-1]  # Default: última coluna
                
                # Procurar menções de colunas na query
                for col in df.columns:
                    if col.lower() in query.lower():
                        target_col = col
                        break

                print(f"🚀 Iniciando treinamento de modelos (target='{target_col}')...")
                
                # Simulação de barra de progresso (5 etapas do pipeline)
                etapas = [
                    "Preparando dados",
                    "Dividindo treino/teste",
                    "Treinando modelos base",
                    "Otimizando hiperparâmetros",
                    "Gerando relatório comparativo"
                ]
                for i, etapa in enumerate(etapas, 1):
                    tqdm.write(f"[{i}/{len(etapas)}] {etapa}...")
                    time.sleep(0.5)  # ⏳ s

                result = train_ml_models(df, target_col)
                print(f"DEGUB ML Traning Tool: {result}")
                return result
            
            def ml_visualization_tool(query: str) -> str:
                """Gráficos de comparação ML"""
                target_col = df.columns[-1]
                for col in df.columns:
                    if col.lower() in query.lower():
                        target_col = col
                        break
                
                create_ml_comparison_plots(df, target_col)
                return "Gráficos de comparação ML criados com sucesso"
            
            tools.extend([
                Tool(
                    name="analise_ml",
                    func=ml_analysis_tool,
                    description="Analisa dados para Machine Learning"
                ),
                Tool(
                    name="treinar_modelos",
                    func=ml_training_tool,
                    description="Treina múltiplos modelos de ML"
                ),
                Tool(
                    name="graficos_ml",
                    func=ml_visualization_tool,
                    description="Cria gráficos de comparação de modelos"
                )
            ])
        else:
            def ml_unavailable(query: str) -> str:
                return "Bibliotecas de ML não disponíveis. Instale: pip install scikit-learn xgboost imbalanced-learn"
            
            tools = [Tool(
                name="aviso_ml",
                func=ml_unavailable,
                description="Avisa sobre bibliotecas ML não disponíveis"
            )]
    
    except ImportError:
        def ml_import_error(query: str) -> str:
            return "Módulo ml_tools não encontrado. Verifique se o arquivo está no local correto."
        
        tools = [Tool(
            name="erro_import_ml",
            func=ml_import_error,
            description="Erro de import do módulo ML"
        )]

    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            """Você é um especialista em Machine Learning e Data Science.

CAPACIDADES:
- Análise de qualidade de dados para ML
- Treinamento automático de múltiplos modelos
- Comparação de performance
- Criação de visualizações

WORKFLOW:
1. Sempre analise os dados primeiro
2. Identifique o tipo de problema (classificação/regressão)
3. Treine múltiplos modelos
4. Compare performances
5. Forneça recomendações

Seja didático e forneça insights acionáveis.
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
        handle_parsing_errors=True
    )


def agente_conclusoes(
    llm: BaseChatModel,
    agente_geral_executor: AgentExecutor,
    agente_graficos_executor: AgentExecutor,
    agente_ml_executor: AgentExecutor = None
) -> AgentExecutor:
    """
    Agente de conclusões: sintetiza resultados de análises e gráficos.
    NÃO chama ferramentas, apenas gera texto.
    Sempre responda em português
    """
    def analise_wrapper(query):
        """Wrapper para o agente geral"""
        try:
            result = agente_geral_executor.invoke({"input": query})
            resposta = result.get("output") or result.get("output_txt") or "Sem resultado da análise"
            print(f"DEBUG Analise Wrapper: {resposta}")
            return resposta
        except Exception as e:
            return f"Erro na análise: {str(e)}"

    def grafico_wrapper(query):
        """Wrapper para o agente de gráficos"""
        try:
            result = agente_graficos_executor.invoke({"input": query})
            # Se o resultado tiver figura, só retornamos mensagem resumida
            if isinstance(result, dict) and "mensagem" in result:
                return result["mensagem"]
            resposta = result.get("output") or result.get("output_text") or "Sem resultado do gráfico"
            print(f"DEBUG Grafico Wrapper: {resposta}")
            return resposta
        except Exception as e:
            return f"Erro no gráfico: {str(e)}"

    def ml_wrapper(query):
        """Wrapper para o agente ML (se disponível)"""
        if agente_ml_executor:
            try:
                result = agente_ml_executor.invoke({"input": query})
                resposta = result.get("output") or result.get("output_text") or "Sem resultado ML"
                print(f"DEBUG Ml Wrapper: {resposta}")
                return resposta
            except Exception as e:
                return f"Erro no ML: {str(e)}"
        else:
            return "Agente de ML não disponível"

    tools = [
        Tool(
            name="analise_dados",
            func=analise_wrapper,
            description="Realiza análise estatística e descritiva dos dados"
        ),
        Tool(
            name="criar_grafico",
            func=grafico_wrapper,
            description="Cria visualizações e gráficos dos dados"
        )
    ]
    
    # Adicionar ferramenta ML se disponível
    if agente_ml_executor:
        tools.append(Tool(
            name="machine_learning",
            func=ml_wrapper,
            description="Executa análises de Machine Learning"
        ))

    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            """Você é um especialista em síntese de análises de dados.
            Sempre responda em português brasileiro
            
            MISSÃO:
            - Leia os resultados já produzidos por outros agentes (estatísticas, gráficos, análises).
            - Combine e resuma em conclusões claras e acionáveis.
            - Identifique padrões, tendências e insights relevantes.
            - Sugira próximos passos de investigação ou decisão.
            
            ⚠️ IMPORTANTE:
            - NÃO invoque ferramentas.
            - Responda apenas em texto natural.
"""
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad") # Se der erro ativar
    ])

    agente = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(
        agent=agente, 
        tools=tools, 
        verbose=True, 
        handle_parsing_errors=True
    )


# Função de teste
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

    # Teste dos agentes
    agente_geral_exec = agente_geral(llm, df)
    agente_graficos_exec = agente_graficos(llm, df)
    agente_ml_exec = agente_machine_learning(llm, df)
    
    # Teste do agente ML
    resultado_ml = agente_ml_exec.invoke({"input": "Analise os dados e treine modelos de machine learning"})
    print("Resultado ML:", resultado_ml.get("output"))
    print()
    print("Resultado ML output_text:", resultado_ml.get("output_text"))