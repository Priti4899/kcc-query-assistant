import faiss
import pickle
import numpy as np

class VectorStore:
    def __init__(self, embed_file):
        with open(embed_file, "rb") as f:
            self.docs, self.embeddings = pickle.load(f)
        self.index = faiss.IndexFlatL2(len(self.embeddings[0]))
        self.index.add(np.array(self.embeddings))

    def search(self, query_embedding, k=3):
        distances, indices = self.index.search(np.array([query_embedding]), k)
        return [self.docs[i] for i in indices[0]], distances[0]


