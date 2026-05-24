import os

os.environ["HF_HOME"]                         = os.path.abspath("data/model_cache")
os.environ["TRANSFORMERS_CACHE"]              = os.path.abspath("data/model_cache")
os.environ["SENTENCE_TRANSFORMERS_HOME"]      = os.path.abspath("data/model_cache")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_OFFLINE"]                 = "1"