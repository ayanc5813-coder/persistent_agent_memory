# =========================================================
# INSTALL DEPENDENCIES
# =========================================================

!pip install -q chromadb sentence-transformers gradio openai tiktoken

# =========================================================
# MOUNT GOOGLE DRIVE
# =========================================================

from google.colab import drive
drive.mount('/content/drive')

# =========================================================
# IMPORTS
# =========================================================

import os
import uuid
import chromadb
import datetime
import gradio as gr

from sentence_transformers import SentenceTransformer
from openai import OpenAI

# =========================================================
# CONFIG
# =========================================================

OPENROUTER_API_KEY = "YOUR_API_KEY"

MODEL_NAME = "nvidia/nemotron-3-super-120b-a12b:free"

BASE_PATH = "/content/drive/MyDrive/agent_memory_system"

# =========================================================
# CREATE FOLDERS
# =========================================================

folders = [
    "chroma_db",
    "summaries",
    "logs"
]

for folder in folders:
    os.makedirs(
        os.path.join(BASE_PATH, folder),
        exist_ok=True
    )

print("Folders ready")

# =========================================================
# EMBEDDING MODEL
# =========================================================

print("Loading embedding model...")

embedding_model = SentenceTransformer(
    "BAAI/bge-small-en-v1.5"
)

print("Embedding model loaded")

# =========================================================
# CHROMADB
# =========================================================

print("Initializing ChromaDB...")

chroma_client = chromadb.PersistentClient(
    path=os.path.join(BASE_PATH, "chroma_db")
)

collection = chroma_client.get_or_create_collection(
    name="agent_memory"
)

print("ChromaDB initialized")

# =========================================================
# OPENROUTER CLIENT
# =========================================================

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY
)

print("OpenRouter connected")

# =========================================================
# SESSION
# =========================================================

SESSION_ID = str(uuid.uuid4())

print("Session ID:", SESSION_ID)

# =========================================================
# SHORT TERM MEMORY
# =========================================================

short_term_memory = []

# =========================================================
# SAVE MEMORY
# =========================================================

def save_memory(user_message, assistant_response):

    try:

        combined_text = f"""
        User: {user_message}

        Assistant: {assistant_response}
        """

        embedding = embedding_model.encode(
            combined_text
        ).tolist()

        memory_id = str(uuid.uuid4())

        metadata = {
            "session_id": SESSION_ID,
            "timestamp": str(datetime.datetime.now())
        }

        collection.add(
            documents=[combined_text],
            embeddings=[embedding],
            metadatas=[metadata],
            ids=[memory_id]
        )

        short_term_memory.append({
            "user": user_message,
            "assistant": assistant_response
        })

        return "Memory saved"

    except Exception as e:

        print("Save memory error:", e)

# =========================================================
# RETRIEVE MEMORIES
# =========================================================

def retrieve_memories(query, n_results=5):

    try:

        query_embedding = embedding_model.encode(
            query
        ).tolist()

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )

        memories = []

        if results["documents"]:

            for doc in results["documents"][0]:

                memories.append(doc)

        return memories

    except Exception as e:

        print("Retrieve memory error:", e)

        return []

# =========================================================
# MEMORY IMPORTANCE
# =========================================================

def score_memory_importance(text):

    keywords = [
        "remember",
        "favorite",
        "important",
        "project",
        "architecture",
        "bug",
        "error",
        "solution"
    ]

    score = 0

    for word in keywords:

        if word.lower() in text.lower():

            score += 1

    return score

# =========================================================
# SUMMARIZE SESSION
# =========================================================

def summarize_session():

    try:

        if len(short_term_memory) == 0:

            return "No memory"

        conversation_text = "\n".join([
            f"User: {m['user']}\nAssistant: {m['assistant']}"
            for m in short_term_memory
        ])

        prompt = f"""
        Summarize this conversation clearly.

        Conversation:

        {conversation_text}
        """

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        summary = response.choices[0].message.content

        summary_path = os.path.join(
            BASE_PATH,
            "summaries",
            f"{SESSION_ID}.txt"
        )

        with open(summary_path, "w") as f:

            f.write(summary)

        return summary

    except Exception as e:

        return f"Summary error: {e}"

# =========================================================
# MAIN AGENT
# =========================================================

def agent_chat(user_input, history=None):

    if history is None:

        history = []

    try:

        retrieved_memories = retrieve_memories(
            user_input
        )

        memory_context = "\n\n".join(
            retrieved_memories
        )

        recent_context = "\n".join([
            f"User: {m['user']}\nAssistant: {m['assistant']}"
            for m in short_term_memory[-5:]
        ])

        system_prompt = f"""
        You are a persistent AI assistant.

        Use long-term memory when useful.

        LONG TERM MEMORY:

        {memory_context}

        RECENT MEMORY:

        {recent_context}
        """

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_input
                }
            ],
            temperature=0.7
        )

        assistant_response = (
            response
            .choices[0]
            .message
            .content
        )

        importance = score_memory_importance(
            user_input
        )

        if importance >= 1:

            save_memory(
                user_input,
                assistant_response
            )

        history.append({
            "role": "user",
            "content": user_input
        })

        history.append({
            "role": "assistant",
            "content": assistant_response
        })

        return history

    except Exception as e:

        print("Agent error:", e)

        history.append({
            "role": "assistant",
            "content": f"Error: {str(e)}"
        })

        return history

# =========================================================
# REFLECTION AGENT
# =========================================================

def reflection_agent():

    try:

        memories = retrieve_memories(
            "mistakes bugs failures"
        )

        joined = "\n".join(memories)

        prompt = f"""
        Analyze these memories.

        Find:
        - recurring mistakes
        - repeated bugs
        - optimization opportunities

        Memories:

        {joined}
        """

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        return (
            response
            .choices[0]
            .message
            .content
        )

    except Exception as e:

        return f"Reflection error: {e}"

# =========================================================
# LOAD MEMORIES
# =========================================================

def load_all_memories():

    try:

        results = collection.get()

        docs = results["documents"]

        print(f"Loaded {len(docs)} memories")

        return docs

    except Exception as e:

        print("Load memory error:", e)

        return []

# =========================================================
# HYBRID SEARCH
# =========================================================

def hybrid_search(query):

    try:

        semantic_results = retrieve_memories(query)

        keyword_results = []

        all_memories = collection.get()["documents"]

        for memory in all_memories:

            if query.lower() in memory.lower():

                keyword_results.append(memory)

        combined = list(
            set(
                semantic_results + keyword_results
            )
        )

        return combined[:5]

    except Exception as e:

        print("Hybrid search error:", e)

        return []

# =========================================================
# GRADIO UI
# =========================================================

with gr.Blocks() as demo:

    gr.Markdown(
        "# Persistent AI Memory Agent"
    )

    chatbot = gr.Chatbot(
        type="messages",
        height=500
    )

    msg = gr.Textbox(
        placeholder="Ask something..."
    )

    clear = gr.Button("Clear Chat")

    state = gr.State([])

    def respond(message, history):

        updated_history = agent_chat(
            message,
            history
        )

        return (
            "",
            updated_history,
            updated_history
        )

    msg.submit(
        respond,
        [msg, state],
        [msg, state, chatbot]
    )

    clear.click(
        lambda: ("", [], []),
        None,
        [msg, state, chatbot],
        queue=False
    )

print("Launching app...")

demo.launch(debug=True)

# =========================================================
# TEST IDEAS
# =========================================================

# Ask:
#
# "My favorite framework is LangGraph"
#
# Then ask:
#
# "What framework do I like?"
#
# =========================================================
