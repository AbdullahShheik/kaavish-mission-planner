"""
llm_client.py

Initializes the Groq LLM client and exposes a single function to call
the model.
"""

from groq import Groq
from planner.config import GROQ_API_KEY, GROQ_MODEL


# Client initialization

def get_client() -> Groq:
    """
    Returns an initialized Groq client using the API key from config.
    """
    return Groq(api_key=GROQ_API_KEY)


# LLM call

def call_llm_chat(client: Groq, messages: list[dict], model: str = None) -> str:
    """
    Sends a list of chat messages to the Groq model and returns the response text.
    Deterministic (temperature=0) with JSON object output format.
    """
    selected_model = model or GROQ_MODEL
    completion = client.chat.completions.create(
        model=selected_model,
        messages=messages,
        temperature=0,
        response_format={"type": "json_object"},
    )
    return completion.choices[0].message.content


def call_llm(client: Groq, system_prompt: str, user_prompt: str) -> str:
    """
    Sends a system + user prompt to the configured Groq model and returns
    the raw response text.

    temperature=0 makes the output deterministic, same input always
    produces the same output.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt},
    ]
    return call_llm_chat(client, messages)