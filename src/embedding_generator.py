from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import pickle

def generate_embeddings(input_csv, output_pkl):
    model = SentenceTransformer("all-mpnet-base-v2")
    df = pd.read_csv("D:\Mypython\kcc_query_assistant\data\cleaned_kcc.csv")
    sentences = df["pair"].tolist()
    embeddings = model.encode(sentences, show_progress_bar=True)
    
    with open(output_pkl, "wb") as f:
        pickle.dump((df.to_dict(orient="records"), embeddings), f)
    print("Embeddings saved.")

if __name__ == "__main__":
    generate_embeddings("D:\Mypython\kcc_query_assistant\data\cleaned_kcc1.csv", "D:\Mypython\kcc_query_assistant\data\kcc_embeddings1.pkl")
