from openai import OpenAI
import os
from dotenv import load_dotenv

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key="OPENAI_API_KEY")

resp = client.chat.completions.create(
    model="gpt-5-mini",
    messages=[{"role": "user", "content": "Say hello"}],
)

print(resp.choices[0].message.content)
