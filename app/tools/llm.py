from openai import OpenAI
from azure.identity import DefaultAzureCredential
from config import AZURE_API_KEY,AZURE_DEPLOYMENT,AZURE_ENDPOINT

class LLM:
    def __init__(self):
        self.client=OpenAI(
            base_url=AZURE_ENDPOINT,
            api_key=AZURE_API_KEY
        )
    
    def chat(self, messages: list) -> str:
        completion = self.client.chat.completions.create(
            model    = AZURE_DEPLOYMENT,
            messages = messages,
        )
        return completion.choices[0].message.content