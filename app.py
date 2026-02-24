from src.data_loader import load_all_documents
from src.vectorstore import FaissVectorStore
from src.search import RAGSearch

import os

# Example usage
if __name__ == "__main__":
    docs = load_all_documents("data")
    store = FaissVectorStore("faiss_store")
    
    if not os.path.exists("faiss_store/faiss.index"):
        print("[INFO] Vector store not found. Building...")
        store.build_from_documents(docs)
    
    store.load()
    #print(store.query("What is attention mechanism?", top_k=3))
    rag_search = RAGSearch()
    query = "What is attention mechanism?"
    summary = rag_search.search_and_summarize(query, top_k=3)
    print("Summary:", summary)