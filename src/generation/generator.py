import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

GROQ_MODEL = "openai/gpt-oss-120b"

def get_groq_client():
    return Groq(api_key=os.getenv("GROQ_API_KEY"))

def build_prompt(query: str, context_chunks: list[str]) -> str:
    context = "\n\n".join(f"[Chunk {i+1}]\n{c}" for i, c in enumerate(context_chunks))
    return f"""Answer the question using ONLY the context below. If the context doesn't contain the answer, say "I don't have enough information to answer that."

Context:
{context}

Questions: {query}

Answer:"""

def generate_answer(client: Groq, query: str, context_chunks: list[str]) -> str:
    prompt = build_prompt(query, context_chunks)
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=500
    )
    return response.choices[0].message.content
