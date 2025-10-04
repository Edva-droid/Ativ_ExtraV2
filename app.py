import json
import re
import streamlit as st
import pandas as pd
from src.agent.agent_orquestrador import criar_orquestrador
from src.agent.config_agent import Settings
from src.data.bases import download_base
from src.agent.eda_utils import analise_eda_completa
from src.util.utils import gerar_hash_df_leve
from langchain_groq import ChatGroq
import matplotlib.pyplot as plt
import traceback
from datetime import datetime
import io
import zipfile

# =============================================================================
# CONFIGURAÇÃO INICIAL
# =============================================================================

st.set_page_config(
    page_title="🤖 EDA Inteligente - Análise de CSV com IA", 
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/seu-usuario/seu-repo',
        'Report a bug': "https://github.com/seu-usuario/seu-repo/issues",
        'About': "Sistema EDA usando LangGraph e IA - Trabalha com qualquer CSV"
    }
)

# =============================================================================
# FUNÇÕES DE CARREGAMENTO DE DADOS
# =============================================================================

def clean_dataframe_for_streamlit(df):
    """
    Limpa o DataFrame para compatibilidade com Streamlit/PyArrow
    """
    df_clean = df.copy()
    
    try:
        # Converter tipos problemáticos
        for col in df_clean.columns:
            # Verificar se é uma coluna problemática
            if df_clean[col].dtype == 'object':
                # Tentar converter para numérico se possível
                try:
                    # Se todos os valores não-nulos são números, converter
                    pd.to_numeric(df_clean[col], errors='raise')
                    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
                except:
                    # Se não conseguir converter, manter como string
                    df_clean[col] = df_clean[col].astype(str)
            
            # Lidar com tipos Int64 problemáticos
            elif str(df_clean[col].dtype) in ['Int64', 'int64']:
                # Converter para int32 ou float se houver NaNs
                if df_clean[col].isna().any():
                    df_clean[col] = df_clean[col].astype('float64')
                else:
                    try:
                        df_clean[col] = df_clean[col].astype('int32')
                    except:
                        df_clean[col] = df_clean[col].astype('float64')
            
            # Converter boolean nullable para bool padrão
            elif str(df_clean[col].dtype) == 'boolean':
                df_clean[col] = df_clean[col].astype('bool')
        
        return df_clean
    
    except Exception as e:
        # Se a limpeza falhar, converter tudo para string como último recurso
        print(f"Erro na limpeza, convertendo para strings: {e}")
        for col in df_clean.columns:
            df_clean[col] = df_clean[col].astype(str)
        return df_clean


def load_csv_file(uploaded_file):
    """Carrega arquivo CSV uploaded"""
    try:
        # Reset do ponteiro do arquivo
        uploaded_file.seek(0)
        
        # Verificar se é um arquivo zip
        if uploaded_file.name.endswith('.zip'):
            with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
                # Listar arquivos no zip
                file_list = zip_ref.namelist()
                csv_files = [f for f in file_list if f.endswith('.csv')]
                
                if not csv_files:
                    st.error("❌ Nenhum arquivo CSV encontrado no ZIP")
                    return None
                
                # Se houver múltiplos CSVs, deixar o usuário escolher
                if len(csv_files) > 1:
                    selected_file = st.selectbox(
                        "📁 Múltiplos CSVs encontrados. Selecione um:",
                        csv_files
                    )
                else:
                    selected_file = csv_files[0]
                
                # Ler o CSV selecionado
                with zip_ref.open(selected_file) as csv_file:
                    df = pd.read_csv(csv_file)
                    
        else:
            # Arquivo CSV direto - tentar diferentes configurações
            encodings_to_try = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
            df = None
            
            for encoding in encodings_to_try:
                try:
                    # Reset do ponteiro antes de cada tentativa
                    uploaded_file.seek(0)
                    
                    # Primeiro, tentar ler normalmente
                    try:
                        df = pd.read_csv(uploaded_file, encoding=encoding)
                        break
                    except:
                        # Se falhar, tentar sem header (caso a primeira linha seja dados)
                        uploaded_file.seek(0)
                        df = pd.read_csv(uploaded_file, encoding=encoding, header=None)
                        
                        # Gerar nomes de colunas se não houver header
                        df.columns = [f'Column_{i+1}' for i in range(len(df.columns))]
                        st.info(f"📋 Arquivo carregado sem cabeçalhos. Colunas nomeadas como: {', '.join(df.columns)}")
                        break
                        
                except UnicodeDecodeError:
                    continue
                except Exception as e:
                    # Log do erro mas continue tentando
                    st.warning(f"Tentativa com encoding {encoding} falhou: {str(e)}")
                    continue
            
            if df is None:
                st.error("❌ Não foi possível ler o arquivo CSV com nenhum encoding testado.")
                return None
        
        # Validar se o DataFrame foi carregado corretamente
        if df is None or df.empty:
            st.error("❌ DataFrame carregado está vazio")
            return None
        
        # Verificar se as colunas têm nomes estranhos (apenas números)
        if all(str(col).isdigit() or col == 0 for col in df.columns):
            st.warning("⚠️ Detectadas colunas com nomes numéricos. Possivelmente o arquivo não tem cabeçalhos apropriados.")
            # Renomear colunas para algo mais legível
            df.columns = [f'Feature_{i+1}' for i in range(len(df.columns))]
            st.info(f"🔄 Colunas renomeadas para: {', '.join(df.columns)}")
        
        # NOVO: Limpar DataFrame para compatibilidade com Streamlit
        df_clean = clean_dataframe_for_streamlit(df)
        
        st.session_state["intermediate_results"] = {}
        # st.session_state["intermediate_results"]["analise_eda"] = eda_resultados
        # Log de sucesso com informações básicas
        st.success(f"📊 Dataset carregado com sucesso: {df_clean.shape[0]} linhas x {df_clean.shape[1]} colunas")
        
        return df_clean
        
    except Exception as e:
        st.error(f"❌ Erro crítico ao carregar arquivo: {str(e)}")
        st.error("💡 Verifique se o arquivo é um CSV válido")
        
        # Debug adicional
        st.error("🔍 Informações de debug:")
        st.code(f"Erro: {type(e).__name__}: {str(e)}")
        return None

@st.cache_data
def get_default_dataframe():
    try:
        df = download_base()
        
        return df
    except Exception as e:
        st.warning(f"⚠️ Não foi possível carregar dataset padrão: {e}")
        return pd.DataFrame()

@st.cache_resource
def get_llm():
    """Inicializa o LLM uma única vez com configurações otimizadas"""
    try:
        return ChatGroq(
            temperature=0,
            # groq_api_key=Settings.GROQ_API_KEY,
            model=Settings.LLM_MODEL,
            max_tokens=4096,
            timeout=60  # Timeout maior para análises complexas
        )
    except Exception as e:
        st.error(f"❌ Erro ao inicializar LLM: {e}")
        return None

@st.cache_resource
def get_orchestrator(_llm, _df_hash):
    """Cria o orquestrador com base no hash do DataFrame"""
    if _llm is None or _df_hash is None:
        return None
    try:
        # Usar o hash para garantir que o cache seja invalidado quando o DF mudar
        return criar_orquestrador(_llm, st.session_state.current_df)
    except Exception as e:
        st.error(f"❌ Erro ao criar orquestrador: {e}")
        return None

# =============================================================================
# INICIALIZAÇÃO DE ESTADO
# =============================================================================

# Inicializar LLM
llm = get_llm()

# Inicializar estado da sessão - COMEÇAR VAZIO
if "messages" not in st.session_state:
    st.session_state.messages = []

if "total_queries" not in st.session_state:
    st.session_state.total_queries = 0

if "session_start" not in st.session_state:
    st.session_state.session_start = datetime.now()

# MODIFICAÇÃO: Inicializar com DataFrame vazio
if "current_df" not in st.session_state:
    st.session_state.current_df = pd.DataFrame()  # Começa vazio

if "dataset_name" not in st.session_state:
    st.session_state.dataset_name = None  # Começa sem dataset

if "analysis_history" not in st.session_state:
    st.session_state.analysis_history = []

# =============================================================================
# INTERFACE PRINCIPAL
# =============================================================================

st.title("🤖 EDA Inteligente - Análise de CSV com IA")
st.markdown("*Sistema genérico para análise exploratória de qualquer arquivo CSV*")

# =============================================================================
# SEÇÃO DE CARREGAMENTO DE DADOS
# =============================================================================

st.header("📁 Carregamento de Dados")

# Exibir instruções se não houver dataset carregado
if st.session_state.current_df.empty:
    st.info("👋 **Bem-vindo ao Sistema EDA Inteligente!**\n\nPara começar, carregue um arquivo CSV ou use o dataset padrão abaixo.")

col1, col2 = st.columns([3, 1])

with col1:
    uploaded_file = st.file_uploader(
        "🔄 Carregue seu arquivo CSV",
        type=['csv'],
        help="Suporta arquivos CSV. O sistema analisará qualquer estrutura de dados."
    )

with col2:
    use_default = st.button("📊 Usar Dataset Padrão", width="stretch")

# Processar carregamento de arquivo
if uploaded_file is not None:
    # Mostrar informações do arquivo antes do processamento
    st.write("**Informações do arquivo:**")
    st.write(f"- Nome: {uploaded_file.name}")
    st.write(f"- Tamanho: {uploaded_file.size:,} bytes ({uploaded_file.size/1024:.1f} KB)")
    st.write(f"- Tipo: {uploaded_file.type}")
    
    # Verificar se o arquivo não é muito grande (limite de 50MB)
    if uploaded_file.size > 50 * 1024 * 1024:
        st.warning("⚠️ Arquivo muito grande (>50MB). O carregamento pode ser lento.")
    
    # Usar progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        status_text.text("🔍 Iniciando carregamento...")
        progress_bar.progress(10)
        
        status_text.text("📈 Processando arquivo CSV...")
        progress_bar.progress(30)
        
        new_df = load_csv_file(uploaded_file)
        progress_bar.progress(70)
        
        if new_df is not None:
            status_text.text("✅ Validando dados...")
            progress_bar.progress(90)
            
            # Debug: verificar se o DataFrame foi carregado
            st.write(f"**DataFrame processado:** {new_df.shape[0]} linhas x {new_df.shape[1]} colunas")
            st.write(f"**Colunas encontradas:** {new_df.columns.tolist()}")
            
            # Atualizar estado da sessão
            st.session_state.current_df = new_df
            st.session_state.dataset_name = uploaded_file.name
            st.session_state.messages = []  # Limpar histórico ao carregar novo dataset
            st.session_state.analysis_history = []
            
            progress_bar.progress(100)
            status_text.text("✅ Carregamento concluído!")
            
            st.success(f"Arquivo '{uploaded_file.name}' carregado com sucesso!")
            
            # Mostrar preview dos dados
            with st.expander("👀 Preview dos dados carregados", expanded=True):
                try:
                    # Mostrar apenas informações básicas se houver problemas de exibição
                    st.write(f"**Shape:** {new_df.shape[0]} linhas × {new_df.shape[1]} colunas")
                    st.write(f"**Colunas:** {', '.join(new_df.columns.tolist()[:10])}{'...' if len(new_df.columns) > 10 else ''}")
                    
                    # Tentar mostrar o dataframe, com fallback para informações básicas
                    try:
                        st.dataframe(new_df.head(), width="stretch")
                    except Exception as display_error:
                        st.warning(f"⚠️ Não foi possível exibir o preview visual: {display_error}")
                        st.write("**Primeiras 5 linhas (formato texto):**")
                        st.text(str(new_df.head()))
                    
                    # Informações dos tipos de dados
                    st.write("**Tipos de dados:**")
                    for col, dtype in new_df.dtypes.items():
                        st.write(f"- **{col}**: {str(dtype)}")
                        
                except Exception as preview_error:
                    st.error(f"Erro no preview: {preview_error}")
                    st.write("Arquivo carregado, mas preview não disponível.")
            
            # # Forçar atualização da interface
            # import time
            # time.sleep(0.5)  # Pequena pausa para garantir que tudo foi processado
            # st.rerun()
            
        else:
            progress_bar.progress(100)
            status_text.text("❌ Falha no carregamento")
            st.error("❌ Não foi possível carregar o arquivo")
            st.info("💡 Tente verificar se o arquivo está no formato CSV correto")
    
    except Exception as e:
        progress_bar.progress(100)
        status_text.text("❌ Erro no processamento")
        st.error(f"Erro inesperado: {str(e)}")
        st.code(f"Erro técnico: {type(e).__name__}: {str(e)}")
    
    finally:
        # Limpar progress bar após 2 segundos
        import time
        time.sleep(2)
        progress_bar.empty()
        status_text.empty()

# Usar dataset padrão
if use_default:
    default_df = get_default_dataframe()
    if not default_df.empty:
        st.session_state.current_df = default_df
        st.session_state.dataset_name = "Dataset Padrão (Credit Card Fraud)"
        st.session_state.messages = []
        st.session_state.analysis_history = []
        st.success("✅ Dataset padrão carregado!")
        st.rerun()

# Obter DataFrame atual
df = st.session_state.current_df
# 99999999
if "current_df" in st.session_state:
    df = st.session_state.current_df

    if df is not None and not df.empty and len(df.columns) > 0:
        current_hash = gerar_hash_df_leve(df)

        if ("eda_summary" not in st.session_state) or (st.session_state.get("eda_hash") != current_hash):
            st.session_state.eda_summary = analise_eda_completa(df)
            st.session_state.eda_hash = current_hash
            st.success("✅ Resumo EDA gerado/atualizado!")

        # st.write(st.session_state.eda_summary)
    else:
        st.warning("⚠️ DataFrame está vazio ou não possui colunas.")
else:
    st.warning("📂 Nenhum DataFrame carregado ainda.")


# =============================================================================
# VALIDAÇÃO E CRIAÇÃO DO ORQUESTRADOR
# =============================================================================

# Criar hash do DataFrame para cache - apenas se não estiver vazio
df_hash = None
if not df.empty:
    df_hash = hash(str(df.shape) + str(df.columns.tolist()) + str(df.iloc[0].tolist() if len(df) > 0 else []))

graph = get_orchestrator(llm, df_hash) if llm is not None and not df.empty else None

# =============================================================================
# SIDEBAR - MOSTRAR APENAS SE HOUVER DATASET
# =============================================================================

with st.sidebar:
    if not df.empty and st.session_state.dataset_name:
        st.header("📊 Informações do Dataset")
        st.write(f"**Dataset Atual:** {st.session_state.dataset_name}")
        
        # Informações básicas
        col1, col2 = st.columns(2)
        with col1:
            st.metric("📏 Linhas", f"{df.shape[0]:.0f}")
        with col2:
            st.metric("📋 Colunas", df.shape[1])
        
        # Tipos de dados detalhados
        st.subheader("🏷️ Análise de Tipos")
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        datetime_cols = df.select_dtypes(include=['datetime']).columns.tolist()
        
        if numeric_cols:
            st.write(f"🔢 **Numéricas ({len(numeric_cols)}):**")
            for col in numeric_cols[:5]:  # Mostrar apenas 5
                st.write(f"  • {col}")
            if len(numeric_cols) > 5:
                st.write(f"  ... e mais {len(numeric_cols) - 5}")
        
        if categorical_cols:
            st.write(f"📝 **Categóricas ({len(categorical_cols)}):**")
            for col in categorical_cols[:5]:
                st.write(f"  • {col}")
            if len(categorical_cols) > 5:
                st.write(f"  ... e mais {len(categorical_cols) - 5}")
        
        if datetime_cols:
            st.write(f"📅 **Data/Hora ({len(datetime_cols)}):**")
            for col in datetime_cols:
                st.write(f"  • {col}")
        
        # Análise de qualidade
        st.subheader("🔍 Qualidade dos Dados")
        null_counts = df.isnull().sum().sum()
        duplicate_counts = df.duplicated().sum()
        
        # Métricas de Variação/Unicidade
        constant_cols_count = (df.nunique() == 1).sum()
        avg_cardinality = df.nunique().mean()
            
        if null_counts > 0:
            st.warning(f"⚠️ {null_counts:,} valores nulos ({(null_counts/df.size*100):.1f}%)")
        else:
            st.success("✅ Sem valores nulos")
        
        if duplicate_counts > 0:
            st.warning(f"🔄 {duplicate_counts:,} linhas duplicadas")
        else:
            st.success("✅ Sem duplicadas")

        if constant_cols_count > 0:
            st.warning(f"⚠️ {constant_cols_count:,} colunas sem variação")
        else:
            st.success("✅ Todas as colunas têm variação")
        
        st.info(f"📈 Cardinalidade média: {avg_cardinality:.1f}")

        # Uso de memória
        memory_usage = df.memory_usage(deep=True).sum() / 1024**2
        if memory_usage > 100:
            st.error(f"💾 {memory_usage:.1f} MB (Alto)")
        elif memory_usage > 10:
            st.warning(f"💾 {memory_usage:.1f} MB (Médio)")
        else:
            st.success(f"💾 {memory_usage:.1f} MB (Baixo)")
        
        # Estatísticas da sessão
        st.divider()
        st.subheader("📈 Estatísticas da Sessão")
        st.write(f"🔢 Consultas realizadas: {st.session_state.total_queries}")
        st.write(f"📋 Análises no histórico: {len(st.session_state.analysis_history)}")
        
        duration = datetime.now() - st.session_state.session_start
        minutes = duration.seconds // 60
        seconds = duration.seconds % 60
        st.write(f"⏱️ Tempo de sessão: {minutes}min {seconds}s")
        
    else:
        # Estado inicial - sem dataset
        st.header("🚀 Comece Aqui!")
        st.write("**Para usar o sistema:**")
        st.write("1. 📤 Carregue um arquivo CSV")
        st.write("2. 📊 Ou use o dataset padrão")
        st.write("3. 💬 Faça perguntas sobre os dados")
        
        st.divider()
        st.subheader("🎯 O que você pode fazer:")
        st.write("• Análise exploratória automatizada")
        st.write("• Detecção de outliers e anomalias") 
        st.write("• Correlações e relações")
        st.write("• Visualizações inteligentes")
        st.write("• Insights e conclusões")
        st.write("• Machine Learning")
    
    # Status do sistema - sempre mostrar
    st.divider()
    st.subheader("🔧 Status do Sistema")
    
    status_llm = "✅" if llm else "❌"
    status_data = "✅" if not df.empty else "⏳"
    status_graph = "✅" if graph else "⏳"
    
    st.write(f"{status_llm} LLM (Groq)")
    st.write(f"{status_data} Dataset")
    st.write(f"{status_graph} Orquestrador")

# =============================================================================
# VALIDAÇÃO DO SISTEMA - MOSTRAR APENAS SE TENTATIVA DE USO SEM DATASET
# =============================================================================

# Verificar se o sistema está pronto para análises
system_ready = all([llm, not df.empty, graph])

# Se não há dataset carregado, mostrar instrução amigável
if df.empty:
    st.warning("📋 **Aguardando carregamento de dados**")
    st.info("Carregue um arquivo CSV ou use o dataset padrão para começar a análise.")
    
    # Mostrar preview do que está disponível
    with st.expander("🔍 Preview das Funcionalidades Disponíveis"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            **📊 Análise Descritiva**
            - Tipos de dados
            - Estatísticas descritivas
            - Distribuições
            - Valores únicos
            """)
        
        with col2:
            st.markdown("""
            **📈 Visualizações**
            - Histogramas
            - Boxplots
            - Correlações
            - Scatter plots
            """)
        
        with col3:
            st.markdown("""
            **🧠 Insights**
            - Detecção de outliers
            - Padrões temporais
            - Conclusões automáticas
            - Recomendações
            """)
    
    st.stop()  # Parar aqui se não há dataset

# Se há problemas no sistema com dataset carregado
if not system_ready and not df.empty:
    st.error("❌ Sistema não está pronto para uso")
    
    if llm is None:
        st.error("🔧 **LLM não inicializado** - Verifique a chave API do Groq")
    
    if graph is None:
        st.error("🔀 **Orquestrador não criado** - Verifique as dependências")
    
    st.info("💡 Corrija os problemas acima para continuar")
    st.stop()

# =============================================================================
# ÁREA DO CHAT INTELIGENTE - APENAS SE DATASET CARREGADO
# =============================================================================

st.header("💬 Chat Inteligente para EDA")

# Container para as mensagens do chat
chat_container = st.container()

with chat_container:
    # Exibir histórico de mensagens
    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Mostrar timestamp para análises importantes
            if message["role"] == "assistant" and len(message["content"]) > 200:
                st.caption(f"🕒 Análise #{i//2 + 1}")

# =============================================================================
# SUGESTÕES INTELIGENTES BASEADAS NOS DADOS
# =============================================================================

# if not st.session_state.messages and not df.empty:  # Apenas se não houver histórico e houver dataset
#     st.info("💡 **Sugestões baseadas no seu dataset:**")
    
#     suggestions = []
    
#     # Sugestões baseadas na estrutura dos dados
#     # Para datasets com muitas colunas numéricas
#     numeric_cols = df.select_dtypes(include=['number']).columns
#     if len(numeric_cols) > 5:
#         suggestions.append("📊 Faça uma análise descritiva completa dos dados")
#         suggestions.append("🔗 Analise a correlação entre as variáveis numéricas")
    
#     # Para datasets com variável target aparente
#     if any('class' in col.lower() for col in df.columns):
#         suggestions.append("🎯 Analise a distribuição da variável target (class)")
#         suggestions.append("🔍 Identifique padrões relacionados à classificação")
    
#     # Para datasets temporais
#     if any('time' in col.lower() or 'date' in col.lower() for col in df.columns):
#         suggestions.append("📈 Analise padrões temporais nos dados")
    
#     # Para detecção de anomalias
#     if len(df) > 1000:
#         suggestions.append("🚨 Detecte outliers e anomalias nos dados")
    
#     # Sugestões gerais
#     suggestions.extend([
#         "📋 Descreva os tipos de dados e estrutura do dataset",
#         "📊 Crie gráficos para visualizar distribuições",
#         "🧠 Gere conclusões inteligentes sobre os dados"
#     ])
    
#     # Mostrar sugestões em colunas
#     cols = st.columns(min(len(suggestions), 3))
#     for i, suggestion in enumerate(suggestions[:6]):  # Máximo 6 sugestões
#         with cols[i % 3]:
#             if st.button(suggestion, key=f"suggestion_{i}", width="stretch"):
#                 # Simular input do usuário
#                 prompt = suggestion.replace("📊", "").replace("🔗", "").replace("🎯", "").replace("🔍", "").replace("📈", "").replace("🚨", "").replace("📋", "").replace("🧠", "").strip()
#                 st.session_state.messages.append({"role": "user", "content": prompt})
#                 st.rerun()

# =============================================================================
# INPUT E PROCESSAMENTO PRINCIPAL
# =============================================================================

if prompt := st.chat_input("💬 Faça sua pergunta sobre os dados (EDA completa, gráficos, conclusões...)"):
    
    # Incrementar contador
    st.session_state.total_queries += 1
    
    # Adicionar à história de análises
    st.session_state.analysis_history.append({
        "timestamp": datetime.now(),
        "query": prompt,
        "dataset": st.session_state.dataset_name
    })
    
    # Adicionar mensagem do usuário ao histórico
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Mostrar mensagem do usuário
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Processar resposta do assistente
    with st.chat_message("assistant"):
        
        # Spinner inteligente baseado no tipo de pergunta
        spinner_texts = {
            "gráfico": "🎨 Criando visualizações...",
            "plot": "📊 Gerando gráficos...",
            "visualização": "🖼️ Preparando visualizações...",
            "correlação": "🔗 Analisando correlações...",
            "outlier": "🚨 Detectando anomalias...",
            "anomalia": "🔍 Identificando outliers...",
            "conclusão": "🧠 Sintetizando insights...",
            "resumo": "📝 Gerando resumo executivo...",
            "análise completa": "🔬 Executando EDA completa...",
            "machine learning": "🤖 Preparando análise ML...",
            "ml": "⚡ Executando pipeline ML..."
        }
        
        spinner_text = "🔍 Analisando dados..."
        for keyword, text in spinner_texts.items():
            if keyword in prompt.lower():
                spinner_text = text
                break
        
        with st.spinner(spinner_text):

            # 1. Inicialize a variável com um valor seguro fora do try
            response_data = {"output": "Erro desconhecido durante o LangGraph."}
            try:

                try:
                    eda_resultados = analise_eda_completa(df)

                    # Garante que sempre exista "resumo"
                    resumo = eda_resultados.get("resumo", {
                        "n_linhas": df.shape[0],
                        "n_colunas": df.shape[1],
                        "variaveis_numericas": list(df.select_dtypes(include=['float64', 'int64']).columns),
                        "variaveis_categoricas": list(df.select_dtypes(include=['object', 'category']).columns),
                        "qtde_num": len(df.select_dtypes(include=['float64', 'int64']).columns),
                        "qtde_cat": len(df.select_dtypes(include=['object', 'category']).columns),
                        "insights": []
                    })

                    prompt_completo = f"""
                    📝 Contexto do Dataset

                    - Linhas: {resumo['n_linhas']}
                    - Colunas: {resumo['n_colunas']}
                    - Variáveis numéricas ({resumo['qtde_num']}): {resumo['variaveis_numericas']}
                    - Variáveis categóricas ({resumo['qtde_cat']}): {resumo['variaveis_categoricas']}
                    - Insights iniciais: {resumo['insights']}

                    ❓ Pergunta do usuário:
                    {prompt}
                    """

                    # Executar o grafo LangGraph
                    response_data = graph.invoke(
                        {
                        "input": prompt_completo,
                        "chat_history": st.session_state.messages[-15:],  # Memória mais ampla
                        "output": "",
                        "intermediate_results": {}
                        },
                        config={
                            "configurable": {
                                "thread_id": f"eda_session_{st.session_state.session_start.timestamp()}"
                            }
                        }
                    )

                except Exception as e:
                    st.error(f"⚠️ Erro ao executar análise EDA: {e}")
                            
                # Extrair resposta final
                final_output = response_data.get("output", "Não consegui gerar uma análise adequada.")
                

                
                # Limpeza e formatação da resposta
                if isinstance(final_output, str):
                    # Remove marcadores de código desnecessários
                    final_output = re.sub(r"^```(?:json|python|text)?\n?", "", final_output)
                    final_output = re.sub(r"\n?```$", "", final_output)
                    final_output = final_output.strip()
                    
                    # Formatação especial para JSON estruturado
                    if final_output.startswith(('{', '[')):
                        try:
                            parsed_json = json.loads(final_output)
                            # Se for um resultado estruturado, formatá-lo melhor
                            if isinstance(parsed_json, dict):
                                formatted_output = ""
                                for key, value in parsed_json.items():
                                    if isinstance(value, dict):
                                        formatted_output += f"\n## {key.replace('_', ' ').title()}\n"
                                        for subkey, subvalue in value.items():
                                            formatted_output += f"**{subkey}:** {subvalue}\n"
                                    else:
                                        formatted_output += f"**{key.replace('_', ' ').title()}:** {value}\n"
                                final_output = formatted_output
                            else:
                                final_output = f"```json\n{json.dumps(parsed_json, ensure_ascii=False, indent=2)}\n```"
                        except json.JSONDecodeError:
                            pass


                # Mostrar resposta formatada
                st.markdown(final_output)
                
                # Verificar e mostrar gráficos matplotlib
                if plt.get_fignums():
                    st.pyplot(plt.gcf())
                    plt.clf()
                
                # Adicionar resposta ao histórico
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": final_output
                })
                
                # Mostrar informações adicionais se a análise foi extensa
                if len(final_output) > 500:
                    st.success("✅ Análise EDA completa realizada!")
                    
                    # Sugerir próximos passos
                    st.info("""
                    💡 **Próximos passos sugeridos:**
                    - Explore aspectos específicos que chamaram atenção
                    - Peça análises de correlação mais detalhadas  
                    - Solicite gráficos específicos para variáveis de interesse
                    - Peça conclusões e insights acionáveis
                    """)
                
                # Informações de debug (expansível)
                with st.expander("🔍 Detalhes da Execução EDA", expanded=False):
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("📝 Análise Solicitada")
                        st.code(prompt, language="text")
                        
                        st.subheader("⚙️ Configuração do Sistema")
                        st.json({
                            "dataset": st.session_state.dataset_name,
                            "linhas": df.shape[0],
                            "colunas": df.shape[1],
                            "modelo": Settings.LLM_MODEL,
                            "temperatura": 0,
                            "memória_chat": len(st.session_state.messages),
                            "tempo_sessão": f"{(datetime.now() - st.session_state.session_start).seconds}s"
                        })
                    
                    with col2:
                        st.subheader("🔄 Resultados Intermediários")
                        intermediate = response_data.get("intermediate_results", {})
                        if intermediate:
                            st.json(intermediate)
                        else:
                            st.info("Processamento direto sem etapas intermediárias")
                        
                        st.subheader("📊 Análise da Resposta")
                        response_stats = {
                            "tamanho_resposta": len(final_output),
                            "tipo_análise": "Complexa" if len(final_output) > 500 else "Simples",
                            "gráficos_gerados": len(plt.get_fignums()) if plt.get_fignums() else 0,
                            "formato": "JSON estruturado" if final_output.startswith(('{', '[')) else "Texto natural"
                        }
                        st.json(response_stats)
                
            except Exception as e:
                # Tratamento robusto de erros
                error_msg = f"❌ **Erro na análise EDA**\n\n`{str(e)}`"
                st.error("Ops! Algo deu errado durante a análise dos dados.")
                
                # Adicionar erro ao histórico
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"Erro na análise: {str(e)}"
                })
                
                # Debug detalhado do erro
                with st.expander("🐛 Informações de Debug"):
                    st.code(f"Tipo do erro: {type(e).__name__}")
                    st.code(f"Mensagem: {str(e)}")
                    st.code("Traceback:")
                    st.code(traceback.format_exc())
                    
                    st.write("**Contexto:**")
                    st.write(f"- Dataset: {st.session_state.dataset_name}")
                    st.write(f"- Shape: {df.shape}")
                    st.write(f"- Consulta: {prompt}")
                
                # Sugestões de solução
                st.info("""
                💡 **Possíveis soluções:**
                - Tente reformular sua pergunta de forma mais específica
                - Verifique se o dataset foi carregado corretamente
                - Para datasets muito grandes, tente análises mais focadas
                - Recarregue a página se o problema persistir
                """)

# =============================================================================
# CONTROLES E UTILITÁRIOS - APENAS SE HOUVER DATASET
# =============================================================================

if not df.empty:
    st.divider()
    st.subheader("🛠️ Controles do Sistema")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        if st.button("🗑️ Limpar Chat", width="stretch"):
            st.session_state.messages = []
            st.success("Chat limpo!")
            st.rerun()

    with col2:
        if st.button("🔄 Resetar Sistema", width="stretch"):
            st.cache_resource.clear()
            st.cache_data.clear()
            st.session_state.messages = []
            st.session_state.analysis_history = []
            st.session_state.total_queries = 0
            st.success("Sistema resetado!")
            st.rerun()

    with col3:
        # Exportar análises
        if st.session_state.messages and st.button("📥 Exportar EDA", width="stretch"):
            eda_export = {
                "dataset_info": {
                    "name": st.session_state.dataset_name,
                    "shape": df.shape,
                    "columns": df.columns.tolist(),
                    "dtypes": df.dtypes.astype(str).to_dict()
                },
                "session_info": {
                    "timestamp": datetime.now().isoformat(),
                    "total_queries": st.session_state.total_queries,
                    "duration_seconds": (datetime.now() - st.session_state.session_start).seconds,
                    "analysis_count": len(st.session_state.analysis_history)
                },
                "chat_history": st.session_state.messages,
                "analysis_history": [
                    {
                        "timestamp": h["timestamp"].isoformat(),
                        "query": h["query"],
                        "dataset": h["dataset"]
                    } for h in st.session_state.analysis_history
                ]
            }
            
            st.download_button(
                label="⬇️ Download Relatório EDA",
                data=json.dumps(eda_export, ensure_ascii=False, indent=2),
                file_name=f"eda_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                width="stretch"
            )

    with col4:
        # Sumário do dataset
        if st.button("📋 Info Dataset", width="stretch"):
            st.info(f"""
            **📊 {st.session_state.dataset_name}**
            
            **Estrutura:**
            - Linhas: {df.shape[0]:,}
            - Colunas: {df.shape[1]}
            - Memória: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB
            
            **Tipos:**
            - Numéricas: {len(df.select_dtypes(include=['number']).columns)}
            - Categóricas: {len(df.select_dtypes(include=['object']).columns)}
            
            **Qualidade:**
            - Nulos: {df.isnull().sum().sum():,}
            - Duplicatas: {df.duplicated().sum():,}
            """)

    with col5:
        # Histórico de análises
        if st.session_state.analysis_history and st.button("📈 Histórico", width="stretch"):
            st.write("**🕒 Histórico de Análises:**")
            for i, analysis in enumerate(st.session_state.analysis_history[-5:], 1):
                st.write(f"{i}. *{analysis['timestamp'].strftime('%H:%M')}* - {analysis['query'][:50]}...")

# =============================================================================
# GUIA DE USO PARA EDA - APENAS SE HOUVER DATASET
# =============================================================================

if not df.empty:
    with st.expander("📖 Guia Completo de EDA - Atendendo aos Requisitos da Banca", expanded=False):
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📋 Descrição dos Dados",
            "📈 Padrões e Tendências", 
            "🚨 Detecção de Outliers",
            "🔗 Relações entre Variáveis",
            "🧠 Conclusões Inteligentes"
        ])
        
        with tab1:
            st.markdown("""
            ## 📊 Descrição dos Dados
            
            **Perguntas que você pode fazer:**
            
            ### Tipos de Dados:
            - "Quais são os tipos de dados (numéricos, categóricos)?"
            - "Mostre a estrutura do dataset"
            - "Descreva as características de cada coluna"
            
            ### Distribuições:
            - "Qual a distribuição de cada variável?"
            - "Crie histogramas das variáveis numéricas"
            - "Mostre a distribuição da variável [nome]"
            
            ### Intervalos e Estatísticas:
            - "Qual o intervalo de cada variável (mínimo, máximo)?"
            - "Quais são as medidas de tendência central?"
            - "Calcule média, mediana para todas as variáveis"
            - "Qual a variabilidade dos dados (desvio padrão, variância)?"
            """)
        
        with tab2:
            st.markdown("""
            ## 📈 Identificação de Padrões e Tendências
            
            **Análises temporais:**
            - "Existem padrões temporais nos dados?"
            - "Analise a evolução temporal da variável Time"
            - "Mostre tendências ao longo do tempo"
            
            **Frequências:**
            - "Quais os valores mais frequentes?"
            - "Identifique valores menos frequentes"
            - "Analise a distribuição de frequências"
            
            **Agrupamentos:**
            - "Existem agrupamentos (clusters) nos dados?"
            - "Identifique padrões de agrupamento"
            - "Analise segmentação natural dos dados"
            """)
        
        with tab3:
            st.markdown("""
            ## 🚨 Detecção de Anomalias (Outliers)
            
            **Identificação:**
            - "Existem valores atípicos nos dados?"
            - "Detecte outliers em todas as variáveis"
            - "Mostre boxplots para identificar anomalias"
            
            **Análise de Impacto:**
            - "Como esses outliers afetam a análise?"
            - "Qual o impacto dos outliers nas estatísticas?"
            
            **Tratamento:**
            - "Os outliers podem ser removidos?"
            - "Sugira tratamento para valores atípicos"
            - "Analise se outliers devem ser investigados"
            """)
        
        with tab4:
            st.markdown("""
            ## 🔗 Relações entre Variáveis
            
            **Correlações:**
            - "Como as variáveis estão relacionadas?"
            - "Existe correlação entre as variáveis?"
            - "Crie matriz de correlação"
            - "Mostre heatmap de correlações"
            
            **Visualizações:**
            - "Crie gráficos de dispersão entre variáveis"
            - "Mostre relações através de scatter plots"
            - "Analise tabelas cruzadas"
            
            **Influências:**
            - "Quais variáveis têm maior influência?"
            - "Identifique variáveis mais importantes"
            - "Analise relação com a variável target"
            """)
        
        with tab5:
            st.markdown("""
            ## 🧠 Conclusões Inteligentes
            
            **Síntese Completa:**
            - "Quais conclusões você obteve dos dados?"
            - "Resuma os principais insights"
            - "Gere relatório executivo"
            
            **Insights Acionáveis:**
            - "Que ações você recomenda baseado na análise?"
            - "Quais são os próximos passos?"
            - "Identifique oportunidades nos dados"
            
            **Análise Preditiva:**
            - "Execute análise completa de machine learning"
            - "Quais modelos são mais adequados?"
            - "Avalie potencial preditivo dos dados"
            """)

# =============================================================================
# EXEMPLOS ESPECÍFICOS PARA DATASETS - APENAS SE HOUVER DADOS
# =============================================================================

if not df.empty:
    
    if "credit" in st.session_state.dataset_name.lower() or "fraud" in st.session_state.dataset_name.lower():
        st.info("""
        💳 **Exemplos específicos para Credit Card Fraud:**
        
        - "Analise a distribuição de fraudes vs transações normais"
        - "Qual a relação entre Amount e Class?"
        - "Existem padrões temporais nas fraudes?"
        - "Como as variáveis V1-V28 (PCA) se relacionam com fraudes?"
        - "Detecte outliers que podem indicar fraudes"
        - "Crie modelos de machine learning para detectar fraudes"
        """)
    else:
        st.info(f"""
        🔍 **Exemplos específicos para seu dataset ({st.session_state.dataset_name}):**
        
        - "Faça uma análise exploratória completa"
        - "Identifique padrões e tendências nos dados"
        - "Detecte outliers e anomalias"
        - "Analise correlações entre variáveis"
        - "Gere insights e conclusões"
        """)

# =============================================================================
# INSTRUÇÕES INICIAIS - APENAS SE NÃO HOUVER DATASET
# =============================================================================

if df.empty:
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📚 Como Funciona")
        st.markdown("""
        **1. Carregue seus dados** 📁
        - Faça upload de qualquer arquivo CSV
        - Ou use nosso dataset de exemplo (Credit Card Fraud)
        
        **2. Faça perguntas naturais** 💬
        - "Quais são os tipos de dados?"
        - "Mostre a distribuição das variáveis"
        - "Existem outliers nos dados?"
        
        **3. Receba análises inteligentes** 🧠
        - Gráficos automáticos
        - Insights acionáveis
        - Conclusões detalhadas
        """)
    
    with col2:
        st.subheader("🎯 Principais Recursos")
        st.markdown("""
        **Análise Automática** 🔍
        - Detecção de tipos de dados
        - Estatísticas descritivas
        - Identificação de padrões
        
        **Visualizações** 📊
        - Histogramas e boxplots
        - Correlações e heatmaps
        - Gráficos personalizados
        
        **Insights Avançados** 🚀
        - Detecção de outliers
        - Machine Learning
        - Recomendações práticas
        """)


# =============================================================================
# RODAPÉ
# =============================================================================

st.divider()
st.markdown(
    """
    <div style='text-align: center; color: #666; font-size: 0.8em;'>
        🎓 <strong>Sistema EDA Genérico - Atividade Banca Examinadora</strong><br>
        ✅ Suporte completo a qualquer CSV • 📊 EDA Automatizada • 🤖 IA Integrada • 🧠 Conclusões Inteligentes<br>
        🔬 Atende todos os requisitos: Descrição • Padrões • Outliers • Correlações • Insights
    </div>
    """, 
    unsafe_allow_html=True
)

