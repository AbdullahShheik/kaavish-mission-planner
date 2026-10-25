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

def call_llm(client: Groq, system_prompt: str, user_prompt: str) -> str:
    """
    Sends a system + user prompt to the configured Groq model and returns
    the raw response text.

    temperature=0 makes the output deterministic, same input always
    produces the same output.
    """
    completion = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )
    return completion.choices[0].message.content