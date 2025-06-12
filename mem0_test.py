import asyncio
from mem0 import AsyncMemory
import os
from mem0.configs.base import MemoryConfig
import google.generativeai as genai
from dotenv import load_dotenv
load_dotenv()

os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")
os.environ["DATABASE_URL"]= os.getenv("DATABASE_URL")
os.environ["SUPABASE_URL"]= os.getenv("SUPABASE_URL")
os.environ["SUPABASE_KEY"]=os.getenv("SUPABASE_KEY")
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

custom_config = MemoryConfig(
    llm={
        "provider": "gemini",
        "config": {
            "model": "gemini-2.0-flash-lite",
            "temperature": 0.1,
            "max_tokens": 5000,
        }
    },
    embedder={
        "provider": "gemini",
        "config": {
            "model": "models/text-embedding-004",
            "embedding_dims": 768
        }
    },
    vector_store={
        "provider": "supabase",
        "config": {
            "connection_string": os.getenv("DATABASE_URL"),
            "collection_name": "memories",
            "embedding_model_dims": 768
        }
    },  
    version="v1.1"

)
memory = AsyncMemory(config=custom_config)

asyncio.run(memory.add(
    messages=[
        {"role": "user", "content": "I'm travelling to SF"},
        {"role": "assistant", "content": "That's great to hear!"}
    ],
    user_id="alice"
))
