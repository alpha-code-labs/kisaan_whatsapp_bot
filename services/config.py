from dotenv import load_dotenv
import os

load_dotenv()

ENV_VARS = [
    "ACCESS_TOKEN",
    "APP_SECRET",
    "VERIFY_TOKEN",
    "REDIS_HOST",
    "REDIS_PORT",
    "GRAPH_API_URL",
    "OPENAI_API_KEY",
    "GEMINI_API_KEY",
    "WEATHER_API_KEY",
    "AZURE_STORAGE_CONNECTION_STRING",
    "AZURE_STORAGE_CONTAINER",
    "CHROMA_DB_DIR",
    "CHROMA_COLLECTION_NAME",
    "RAG_KB_DIR",
]

class Config:
    app_secret = os.getenv("APP_SECRET", "")
    access_token = os.getenv("ACCESS_TOKEN", "")
    verify_token = os.getenv("VERIFY_TOKEN", "")

    port = int(os.getenv("PORT", "8080"))
    redis_host = os.getenv("REDIS_HOST", "localhost")
    redis_port = int(os.getenv("REDIS_PORT", "6379"))
    graph_api_url = os.getenv("GRAPH_API_URL", "https://graph.facebook.com/v24.0")
    openai_api_key = os.getenv("OPENAI_API_KEY", "")
    gemini_api_key = os.getenv("GEMINI_API_KEY", "")
    weather_api_key = os.getenv("WEATHER_API_KEY", "")
    azure_storage_connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING", "")
    azure_storage_container = os.getenv("AZURE_STORAGE_CONTAINER", "")
    _base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    _data_dir_env = os.getenv("DATA_DIR")
    if _data_dir_env:
        data_dir = _data_dir_env if os.path.isabs(_data_dir_env) else os.path.join(_base_dir, _data_dir_env)
    else:
        data_dir = os.path.join(_base_dir, "data")

    _chroma_dir_env = os.getenv("CHROMA_DB_DIR")
    if _chroma_dir_env:
        chroma_db_dir = _chroma_dir_env if os.path.isabs(_chroma_dir_env) else os.path.join(_base_dir, _chroma_dir_env)
    else:
        chroma_db_dir = os.path.join(data_dir, "chroma_db")

    chroma_collection_name = os.getenv("CHROMA_COLLECTION_NAME", "crop_knowledge_base")

    _rag_kb_dir_env = os.getenv("RAG_KB_DIR")
    if _rag_kb_dir_env:
        rag_kb_dir = _rag_kb_dir_env if os.path.isabs(_rag_kb_dir_env) else os.path.join(_base_dir, _rag_kb_dir_env)
    else:
        rag_kb_dir = os.path.join(data_dir, "gemini_responses")

    @staticmethod
    def check_env_variables():
        for key in ENV_VARS:
            if not os.getenv(key):
                print(f"WARNING: Missing the environment variable {key}")

    @staticmethod
    def print_config():
        print("Config values:")
        print(f"app_secret={Config.app_secret}")
        print(f"access_token={Config.access_token}")
        print(f"verify_token={Config.verify_token}")
        print(f"port={Config.port}")
        print(f"redis_host={Config.redis_host}")
        print(f"redis_port={Config.redis_port}")
        print(f"graph_api_url={Config.graph_api_url}")
        print(f"openai_api_key={Config.openai_api_key}")
        print(f"gemini_api_key={Config.gemini_api_key}")
        print(f"weather_api_key={Config.weather_api_key}")
        print(f"data_dir={Config.data_dir}")
        print(f"chroma_db_dir={Config.chroma_db_dir}")
        print(f"chroma_collection_name={Config.chroma_collection_name}")
        print(f"rag_kb_dir={Config.rag_kb_dir}")
        print(f"blob_string={Config.azure_storage_connection_string}")
        print(f"blob_container={Config.azure_storage_container}")
