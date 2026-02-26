from typing import List, Dict, Any, Tuple

from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM

from get_embedding_function import get_embedding_function

import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_PATH = os.path.join(BASE_DIR, "chroma")

PROMPT_TEMPLATE = """
Tu réponds uniquement en français.

Utilise uniquement le contexte fourni (tu peux traduire si le contexte est en anglais).
À partir du contexte fourni, liste TOUTES les options possibles.
Si une option n’est pas dans le contexte, ne l’invente pas.
Si le contexte ne contient pas la réponse, répond exactement : "Je ne sais pas."

Contexte :
{context}

Question : {question}

Réponse :
"""

def query_rag(
    query_text: str,
    k: int = 5,
    model_name: str = "qwen2.5:1.5b",
    source_filter: str = "",
) -> Tuple[str, List[Dict[str, Any]]]:

    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    #DEBUG	
    items = db.get()
    print("[DEBUG] Example source values:")
    print(items["metadatas"][:3])

    for doc in db.get()["metadatas"][:5]:
        print(doc.get("source"))
    #END DEBUG
    try:
        n = len(db.get(include=[])["ids"])
        print(f"[DEBUG] Chroma docs count: {n} | path={CHROMA_PATH}")
    except Exception as e:
        print(f"[DEBUG] Chroma get() failed: {e} | path={CHROMA_PATH}")

    items = db.get(include=[])
    print("[DEBUG] DB ids count:", len(items["ids"]))

    #if source_filter:
    #    results = db.similarity_search_with_score(
    #        query_text, k=k, filter={"source": {"$contains": "monopoly"}}
    #    )
    #else:
    #    results = db.similarity_search_with_score(query_text, k=k)

    source_filter = (source_filter or "").strip()
    query_text = (query_text or "").strip()
    print(f"[DEBUG] query='{query_text}' source_filter='{source_filter}'")

    if source_filter:
        results = db.similarity_search_with_score(query_text, k=k, filter={"source": source_filter})
    else:
        results = db.similarity_search_with_score(query_text, k=k)

    print(f"[DEBUG] Retrieved results: {len(results)}")
    if results:
        for doc, score in results[:3]:
            print("[DEBUG] source=", doc.metadata.get("source"), "page=", doc.metadata.get("page"), "score=", score)
            print("[DEBUG] excerpt=", doc.page_content[:180].replace("\n", " "), "...\n")

    if not results:
        return "Je ne sais pas.", []

    k = min(k, 5)
    MAX_CHARS_PER_CHUNK = 1200  # ajuste 800–2000 profondeur de recherche de réponse

    context_text = "\n\n---\n\n".join(
        [doc.page_content[:MAX_CHARS_PER_CHUNK] for doc, _score in results[:k]]
    )

    print("[DEBUG] context length:", len(context_text))
    print("[DEBUG] calling LLM...")

    prompt = PROMPT_TEMPLATE.format(
        context=context_text,
        question=query_text,
    )

    print("[DEBUG] prompt head:", prompt[:300].replace("\n", " "), "...")

    #llm = OllamaLLM(model=model_name)
    llm = OllamaLLM(model= model_name, num_predict=220, temperature=0.2)
    
    import time
    t0 = time.time()
    print(f"[DEBUG] LLM model={model_name} num_predict=160")

    response_text = llm.invoke(prompt)
    print(f"[DEBUG] LLM done in {time.time()-t0:.2f}s")

    print("[DEBUG] LLM returned")

    EXCERPT_CHARS = 260

    sources = []
    for doc, score in results:
        sources.append({
            "source": doc.metadata.get("source", "unknown_source"),
            "page": doc.metadata.get("page", "unknown_page"),
            "id": doc.metadata.get("id", "no_id"),
            "score": float(score),
            "excerpt": (doc.page_content or "")[:EXCERPT_CHARS].replace("\n", " "),
        })

    return response_text, sources
