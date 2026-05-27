import os
from dotenv import load_dotenv

load_dotenv()

os.environ["HF_HOME"]                         = os.path.abspath("data/model_cache")
os.environ["TRANSFORMERS_CACHE"]              = os.path.abspath("data/model_cache")
os.environ["SENTENCE_TRANSFORMERS_HOME"]      = os.path.abspath("data/model_cache")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_OFFLINE"]                 = "1"
os.environ["OPENAI_API_KEY"]  = os.getenv("AZURE_OPENAI_API_KEY")
os.environ["OPENAI_BASE_URL"] = os.getenv("AZURE_OPENAI_ENDPOINT")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"]    = "your_key"
os.environ["LANGCHAIN_PROJECT"]    = "sciagent"