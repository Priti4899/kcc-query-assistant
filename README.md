# 🌾 KCC Query Assistant

The **KCC Query Assistant** is a local-first, offline-capable AI application designed to help users query agricultural advice using the public **Kisan Call Center (KCC)** dataset. It leverages a locally deployed Large Language Model (LLM) and retrieval-augmented generation (RAG) to deliver intelligent, contextual responses to natural language queries.

---

## 🧩 Problem Statement

Many farmers and agricultural stakeholders seek reliable, contextual, and region-specific advice from the Kisan Call Center (KCC). However, the dataset is often underutilized due to lack of easy access and intelligent query handling. This project builds an AI-powered assistant that:
- Works **offline** using **local LLMs** (via Ollama)
- Allows users to **ask natural-language questions**
- Retrieves answers from the **KCC dataset** using **semantic search**

---

## 🎯 Objectives

1. **Data Integration & Preprocessing**
   - Clean, normalize, and chunk Q&A pairs from KCC data.
   - Export raw and preprocessed data with metadata.

2. **Local LLM Deployment**
   - Run an open-source model (e.g., Gemma 2B) via Ollama.
   - Ensure complete offline capability with CPU/GPU quantization.

3. **Retrieval-Augmented Generation**
   - Generate sentence embeddings using `sentence-transformers`.
   - Store and search vectors using FAISS or ChromaDB.

4. **User Interface**
   - Simple web interface with Streamlit.
   - Clearly differentiate between KCC-sourced answers and fallback (live) search results.

---

## 🚀 How to Run Locally

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/kcc-query-assistant.git
cd kcc-query-assistant
