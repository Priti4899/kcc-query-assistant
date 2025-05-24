import streamlit as st
from rag_pipeline import RAGPipeline

st.set_page_config(page_title="KCC Query Assistant", layout="wide")
st.title("🌾 KCC Query Assistant")

query = st.text_input("Ask your agricultural question:")
rag = RAGPipeline("D:\Mypython\kcc_query_assistant\data\kcc_embeddings.pkl")

if query:
    with st.spinner("Thinking..."):
        answer = rag.ask(query)
    st.markdown("### Answer")
    st.write(answer)
else:
    print("No responce")
