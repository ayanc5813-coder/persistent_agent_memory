# Persistent AI Memory Agent

![Python](https://img.shields.io/badge/Python-3.10-blue)
![ChromaDB](https://img.shields.io/badge/VectorDB-ChromaDB-green)
![OpenRouter](https://img.shields.io/badge/LLM-OpenRouter-orange)
![License](https://img.shields.io/badge/License-MIT-red)
![Status](https://img.shields.io/badge/Status-Experimental-yellow)

A production-oriented persistent memory AI assistant with:

* semantic memory retrieval
* long-term memory
* reflection agents
* hybrid search
* session summarization
* vector database persistence

Built using:

* ChromaDB
* Sentence Transformers
* OpenRouter
* Gradio
* Streamlit

---

# Features

* Persistent vector memory
* Semantic retrieval
* Reflection agent
* Hybrid search
* Session summarization
* Gradio UI
* Streamlit UI
* HuggingFace Spaces compatible
* Google Colab compatible

---

# System Architecture

```text
User
  ↓
Frontend (Gradio / Streamlit)
  ↓
Main Agent
  ↓
Memory Retrieval Layer
  ├── Short-Term Memory
  ├── Semantic Search
  └── Hybrid Search
  ↓
Context Builder
  ↓
LLM (OpenRouter)
  ↓
Response Generation
  ↓
Memory Scoring
  ↓
Persistent Storage (ChromaDB)
```

---

# Installation

```bash
git clone https://github.com/yourusername/persistent-ai-memory-agent.git

cd persistent-ai-memory-agent

pip install -r requirements.txt
```

---

# Environment Variables

Create `.env`

```env
OPENROUTER_API_KEY=your_key_here
MODEL_NAME=nvidia/nemotron-3-super-120b-a12b:free
```

---

# Run Gradio Version

```bash
python app/ui/gradio_app.py
```

---

# Run Streamlit Version

```bash
streamlit run app/ui/streamlit_app.py
```

---

# Future Roadmap

* Agentic workflows
* LangGraph integration
* Autonomous planning
* Tool calling
* Memory compression
* Reranking pipelines
* Multi-agent collaboration
* Local LLM support
* GPU acceleration

---

# License

MIT License
