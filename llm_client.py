# src/llm_client.py
import os
from groq import Groq




def call_groq(system_prompt: str, user_prompt: str, model: str = None) -> str:
"""Call Groq chat completions synchronously.
Returns the assistant's text.
"""
model = model or os.environ.get("GROQ_MODEL")
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
messages = [
{"role": "system", "content": system_prompt},
{"role": "user", "content": user_prompt},
]
resp = client.chat.completions.create(messages=messages, model=model)
# extract generated text
try:
content = resp.choices[0].message.content
except Exception:
# fallback structure
content = resp.choices[0].get('message', {}).get('content', '')
return content