import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

from ragas import EvaluationDataset, evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import Faithfulness, ResponseRelevancy

from src.generation.pipeline import rag_query

load_dotenv()

def get_ragas_llm():
    gemini_llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0,
    )
    return LangchainLLMWrapper(gemini_llm)

def get_ragas_embeddings():
    hf_embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    return LangchainEmbeddingsWrapper(hf_embeddings)

EVAL_QUESTIONS = [
    "What is the attention mechanism in transformers?",
    "What replaces recurrence in the Transformer architecture?",
    "How many attention heads does the Transformer use?",
]

if __name__ == "__main__":
    rows = []
    for question in EVAL_QUESTIONS:
        answer, results = rag_query(question)
        contexts = [hit.payload["text"] for hit in results]
        rows.append({
            "user_input": question,
            "response": answer,
            "retrieved_contexts": contexts,
        })

    dataset = EvaluationDataset.from_list(rows)

    result = evaluate(
        dataset=dataset,
        metrics=[Faithfulness(), ResponseRelevancy()],
        llm=get_ragas_llm(),
        embeddings=get_ragas_embeddings(),
    )

    print(result)
    print(result.to_pandas().to_string())

    # print("\nOverall Scores:")
    # print(result)

    # df = result.to_pandas()

    # print("\nDetailed Results:")
    # print(
    #     df[
    #         ["user_input", "faithfulness", "answer_relevancy"]
    #     ].to_string(index=False)
    # )