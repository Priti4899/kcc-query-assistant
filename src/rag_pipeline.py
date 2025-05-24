import requests
from sentence_transformers import SentenceTransformer
from vector_store import VectorStore

class RAGPipeline:
    def __init__(self, embed_file):
        self.store = VectorStore(embed_file)
        self.embedder = SentenceTransformer("all-mpnet-base-v2")

    def ask(self, query, threshold=0.6):
        q_vec = self.embedder.encode([query])[0]
        docs, distances = self.store.search(q_vec)

        relevant_docs = [d["pair"] for i, d in enumerate(docs) if distances[i] < threshold]
        if not relevant_docs:
            return "No relevant context found locally. You may need internet fallback."

        context = "\n\n".join(relevant_docs[:3])
        prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
        return self.query_llm(prompt)

    def query_llm(self, prompt):
        res = requests.post("http://localhost:11434/api/generate", json={
            "model": "gemma:2b",  # Replace with your Ollama model
            "prompt": prompt,
            "stream": False
        })
        return res.json().get("response", "")

# Test code
if __name__ == "__main__":
    rag = RAGPipeline("D:\Mypython\kcc_query_assistant\data\kcc_embeddings.pkl")
    print(rag.ask("asking about the control measure for aphid infestation in mustard crops"))
