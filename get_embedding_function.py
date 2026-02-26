#from langchain_community.embeddings.ollama import OllamaEmbeddings
#from langchain_community.embeddings.bedrock import BedrockEmbeddings
from langchain_ollama import OllamaEmbeddings

def get_embedding_function():
    # Ollama écoute généralement sur http://localhost:11434
    return OllamaEmbeddings(model="nomic-embed-text")

#def get_embedding_function():
#    embeddings = BedrockEmbeddings(
#        credentials_profile_name="default", region_name="us-east-1"
#    )
    # embeddings = OllamaEmbeddings(model="nomic-embed-text")
#    return embeddings
