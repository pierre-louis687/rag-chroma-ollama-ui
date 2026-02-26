import argparse

from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM

from get_embedding_function import get_embedding_function

CHROMA_PATH = "chroma"
# PROMPT_TEMPLATE = """
# Answer the question based only on the following context.
# If the context is insufficient, say you don't know.

# Context:
# {context}

# Question: {question}
# """
PROMPT_TEMPLATE = """
Tu es un assistant qui répond UNIQUEMENT en français.

Règles strictes :
- Utilise seulement les informations présentes dans le contexte.
- Si la réponse n’est pas dans le contexte, répond exactement : "Je ne sais pas."

Réponse attendue :
- sous forme de liste claire
- vocabulaire des règles officielles de Monopoly
- pas d’anglicismes

Contexte :
{context}

Question : {question}

Réponse (en français) :
"""



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    parser.add_argument("--k", type=int, default=5, help="Number of chunks to retrieve.")
    parser.add_argument("--model", type=str, default="mistral", help="Ollama LLM model.")
    parser.add_argument("--debug", action="store_true", help="Print retrieved context.")
    parser.add_argument("--source", type=str, default="", help="Filter by metadata source (e.g. data/monopoly.pdf)")
    args = parser.parse_args()

    #query_rag(args.query_text, k=args.k, model_name=args.model, debug=args.debug)
    query_rag(
        args.query_text,
        k=args.k,
        model_name=args.model,
        debug=args.debug,
        source_filter=args.source,
    )



# def query_rag(query_text: str, k: int = 5, model_name: str = "mistral", debug: bool = False):
def query_rag(
    query_text: str,
    k: int = 5,
    model_name: str = "mistral",
    debug: bool = False,
    source_filter: str = "",
):

    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    # results = db.similarity_search_with_score(query_text, k=k)
    if source_filter:
        results = db.similarity_search_with_score(
        query_text,
        k=k,
        filter={"source": source_filter},
    )
    else:
        results = db.similarity_search_with_score(query_text, k=k)


    if not results:
        print("No results found in the vector database.")
        return ""

    context_blocks = []
    pretty_sources = []

    for doc, score in results:
        source = doc.metadata.get("source", "unknown_source")
        page = doc.metadata.get("page", "unknown_page")
        chunk_id = doc.metadata.get("id", "no_id")

        context_blocks.append(doc.page_content)
        pretty_sources.append(
            {
                "source": source,
                "page": page,
                "score": score,
                "id": chunk_id,
            }
        )

    context_text = "\n\n---\n\n".join(context_blocks)

    if debug:
        print("\n=== RETRIEVED CONTEXT (DEBUG) ===\n")
        print(context_text)
        print("\n=== END CONTEXT ===\n")

    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE).format(
        context=context_text,
        question=query_text,
    )

    llm = OllamaLLM(model=model_name)
    response_text = llm.invoke(prompt)

    print("\n=== RESPONSE ===\n")
    print(response_text)

    print("\n=== SOURCES ===\n")
    for i, s in enumerate(pretty_sources, start=1):
        print(
            f"{i}. {s['source']} | page={s['page']} | score={s['score']:.4f} | id={s['id']}"
        )

    return response_text


if __name__ == "__main__":
    main()
