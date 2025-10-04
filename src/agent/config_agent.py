#src\agent\config_agent.py
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

class Settings:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    LLM_MODEL = "openai/gpt-oss-20b"
    ADMIN_GRAPHIC_LLM_MODEL = "deepseek-r1-distill-llama-70b"

    # Modelos disponíveis para os executores
    MODELS = [
        "gemma2-9b-it",
        "llama-3.1-8b-instant",
        "llama-3.3-70b-versatile",
        "deepseek-r1-distill-llama-70b"
        "openai/gpt-oss-20b"
        "openai/gpt-oss-120b"
        "groq/compound"
        "meta-llama/llama-4-maverick-17b-128e-instruct"
        "meta-llama/llama-guard-4-12b"

    ]


    @classmethod
    def create_llm(cls):
        """Cria e retorna o LLM configurado com Groq."""
        return ChatGroq(
            temperature=0,
            # groq_api_key=cls.GROQ_API_KEY,
            model=cls.LLM_MODEL
        )

#Verificar se a chave existe
if Settings.GROQ_API_KEY is None:
    raise ValueError("A variavel de ambiente GROQ_API_KEY Não está definida. Verifique se o arquivo .env existe ")
