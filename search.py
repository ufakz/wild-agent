import os
import dotenv

dotenv.load_dotenv()

from xai_sdk import Client
from xai_sdk.chat import user
from xai_sdk.search import SearchParameters

client = Client(api_key=os.getenv("XAI_API_KEY"))

chat = client.chat.create(
    model="grok-4-fast-reasoning",
    search_parameters=SearchParameters(
        mode="auto",
        return_citations=True,
    ),
)
chat.append(user("Provide me a digest of world news on July 9, 2025."))

response = chat.sample()
print(response.content)
print(response.citations)