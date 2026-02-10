from dotenv import load_dotenv
import os
import json

load_dotenv()

ENV_VARS = [
    "ACCESS_TOKEN",
    "APP_SECRET",
    "VERIFY_TOKEN",
    "REDIS_HOST",
    "REDIS_PORT",
    "REDIS_PASSWORD",  # Add this
    "REDIS_SSL",       # Add this
    "GRAPH_API_URL",
    "OPENAI_API_KEY",
    "GEMINI_API_KEY",
    "WEATHER_API_KEY",
    "AZURE_STORAGE_CONNECTION_STRING",
    "AZURE_STORAGE_CONTAINER",
    "CHROMA_DB_DIR",
    "CHROMA_COLLECTION_NAME",
    "RAG_KB_DIR",

    # ---- NEW CHROMA SERVER VARS ----
    "CHROMA_HOST",
    "CHROMA_PORT",
    "CHROMA_TENANT",
    "CHROMA_DATABASE",
]

class Config:
    app_secret = os.getenv("APP_SECRET", "")
    access_token = os.getenv("ACCESS_TOKEN", "")
    verify_token = os.getenv("VERIFY_TOKEN", "")

    port = int(os.getenv("PORT", "8080"))
    redis_host = os.getenv("REDIS_HOST", "localhost")
    redis_port = int(os.getenv("REDIS_PORT", "6379"))
    redis_password = os.getenv("REDIS_PASSWORD", "") 
    # Default to True for AMR, but allow False for local dev
    redis_ssl = os.getenv("REDIS_SSL", "true").lower() == "true"
    graph_api_url = os.getenv("GRAPH_API_URL", "https://graph.facebook.com/v24.0")
    openai_api_key = os.getenv("OPENAI_API_KEY", "")
    gemini_api_key = os.getenv("GEMINI_API_KEY", "")
    weather_api_key = os.getenv("WEATHER_API_KEY", "")
    azure_storage_connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING", "")
    azure_storage_container = os.getenv("AZURE_STORAGE_CONTAINER", "")

    _base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    _data_dir_env = os.getenv("DATA_DIR")
    if _data_dir_env:
        data_dir = (
            _data_dir_env
            if os.path.isabs(_data_dir_env)
            else os.path.join(_base_dir, _data_dir_env)
        )
    else:
        data_dir = os.path.join(_base_dir, "data")

    _sessions_dir_env = os.getenv("SESSIONS_DIR")
    if _sessions_dir_env:
        sessions_dir = (
            _sessions_dir_env
            if os.path.isabs(_sessions_dir_env)
            else os.path.join(_base_dir, _sessions_dir_env)
        )
    else:
        sessions_dir = os.path.join(_base_dir, "sessions")
    # -------- LOCAL CHROMA PATH (only for dev / fallback) --------
    _chroma_dir_env = os.getenv("CHROMA_DB_DIR")
    if _chroma_dir_env:
        chroma_db_dir = (
            _chroma_dir_env
            if os.path.isabs(_chroma_dir_env)
            else os.path.join(_base_dir, _chroma_dir_env)
        )
    else:
        chroma_db_dir = os.path.join(data_dir, "chroma_db")

    chroma_collection_name = os.getenv("CHROMA_COLLECTION_NAME", "crop_knowledge_base")

    _rag_kb_dir_env = os.getenv("RAG_KB_DIR")
    if _rag_kb_dir_env:
        rag_kb_dir = (
            _rag_kb_dir_env
            if os.path.isabs(_rag_kb_dir_env)
            else os.path.join(_base_dir, _rag_kb_dir_env)
        )
    else:
        rag_kb_dir = os.path.join(data_dir, "gemini_responses")

    # -------- NEW: CHROMA SERVER CONFIG --------

    chroma_host = os.getenv("CHROMA_HOST", "localhost")
    chroma_port = int(os.getenv("CHROMA_PORT", "8000"))

    chroma_tenant = os.getenv("CHROMA_TENANT", "default_tenant")
    chroma_database = os.getenv("CHROMA_DATABASE", "default_database")

    # Optional SSL and headers (for future auth/reverse proxy)
    chroma_ssl = os.getenv("CHROMA_SSL", "false").lower() == "true"

    # If you ever need auth headers for Chroma behind a gateway
    chroma_headers = json.loads(os.getenv("CHROMA_HEADERS", "{}"))

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

        # ---- New server values ----
        print(f"chroma_host={Config.chroma_host}")
        print(f"chroma_port={Config.chroma_port}")
        print(f"chroma_tenant={Config.chroma_tenant}")
        print(f"chroma_database={Config.chroma_database}")
        print(f"chroma_ssl={Config.chroma_ssl}")
