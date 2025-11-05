import os
from sentence_transformers import SentenceTransformer
embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# os.environ["HF_HOME"] = "/tmp/hf_home"
# os.environ["HF_HUB_CACHE"] = "/tmp/hf_home"
# os.environ["SENTENCE_TRANSFORMERS_HOME"] = "/tmp/st_cache"

# os.makedirs("/tmp/hf_home", exist_ok=True)
# os.makedirs("/tmp/st_cache", exist_ok=True)

import re
import pickle
import faiss
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from llama_cpp import Llama
from huggingface_hub import hf_hub_download
from supabase import create_client, Client

# ============================================
# Environment Setup
# ============================================

os.environ["OMP_NUM_THREADS"] = str(os.cpu_count())
os.environ["OMP_WAIT_POLICY"] = "PASSIVE"
os.environ["LLAMA_CPP_USE_MLOCK"] = "1"
os.environ["LLAMA_CPP_USE_MMAP"] = "1"

# ============================================
# Llama Model Wrapper
# ============================================
class LlamaWrapper:
    def __init__(self, model_path, n_ctx=2048, n_threads=os.cpu_count()):
        self.llm = Llama(model_path=model_path, n_ctx=n_ctx, n_threads=n_threads)

    def __call__(self, prompt, max_tokens=256, stop=None):
        if hasattr(prompt, "to_string"):
            prompt = prompt.to_string()
        elif isinstance(prompt, dict) and "text" in prompt:
            prompt = prompt["text"]
        elif not isinstance(prompt, str):
            prompt = str(prompt)

        response = self.llm(prompt=prompt, max_tokens=max_tokens, stop=stop)
        return response["choices"][0]["text"].strip()

model_path = hf_hub_download(
    repo_id="Omkar1803/mistral-7b-gguf",
    filename="mistral-7b-instruct-v0.2.Q4_K_M.gguf"
)

llm = LlamaWrapper(
    model_path= model_path
)

# ============================================
# FAISS + Metadata Load
# ============================================
FAISS_DIR = "faiss_db"
INDEX_FILE = os.path.join(FAISS_DIR, "index.faiss")
METADATA_FILE = os.path.join(FAISS_DIR, "metadata.pkl")

if not os.path.exists(FAISS_DIR):
    os.makedirs(FAISS_DIR)

if os.path.exists(INDEX_FILE) and os.path.exists(METADATA_FILE):
    index = faiss.read_index(INDEX_FILE)
    with open(METADATA_FILE, "rb") as f:
        metadata = pickle.load(f)
    chunks = metadata.get("chunks", [])
    sources = metadata.get("sources", [])
else:
    index = faiss.IndexFlatL2(384)
    chunks, sources = [], []

# ============================================
# Supabase Setup
# ============================================
SUPABASE_URL = "https://ghvwpalyqvovyxajdgfb.supabase.co"  
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImdodndwYWx5cXZvdnl4YWpkZ2ZiIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc2MDUzNDA5MSwiZXhwIjoyMDc2MTEwMDkxfQ.jE1dZRKvZT36EF3hCQ9N2-5viromWqbJP_1vnKSwU7Q"             # 🔹 replace
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ============================================
# Utilities
# ============================================
embedding_cache = {}

def get_embedding(text):
    if text not in embedding_cache:
        embedding_cache[text] = embed_model.encode([text]).astype("float32")
    return embedding_cache[text]

def get_similar_chunks(query, top_k=5):
    if not chunks:
        return []
    q_emb = get_embedding(query)
    D, I = index.search(q_emb, min(top_k, len(chunks)))
    return [(chunks[i], sources[i]) for i in I[0]]


def is_greeting(text: str) -> bool:
    greetings = ["hi", "hello", "hey", "hii", "hlo", "yo", "hola"]
    text = text.lower().strip()
    # Match whole words only
    return any(re.search(rf"\b{greet}\b", text) for greet in greetings)

# ============================================
# Supabase Helpers
# ============================================
def get_previous_response(query):
    try:
        result = supabase.table("chat_history").select("bot_response").ilike("user_query", f"%{query}%").execute()
        if result.data:
            return result.data[-1]["bot_response"]
    except Exception as e:
        print("⚠️ Error fetching from Supabase:", e)
    return None

def add_chat_to_faiss(query, response):
    try:
        # Avoid duplicates
        if query in chunks:
            return

        new_emb = embed_model.encode([query]).astype("float32")
        index.add(new_emb)
        chunks.append(response)
        sources.append("supabase_chat")

        faiss.write_index(index, INDEX_FILE)
        with open(METADATA_FILE, "wb") as f:
            pickle.dump({"chunks": chunks, "sources": sources}, f)

        print(f"✅ Added chat to FAISS: '{query[:50]}...'")
    except Exception as e:
        print("⚠️ Failed to update FAISS with chat:", e)

def sync_supabase_history_to_faiss():
    try:
        result = supabase.table("chat_history").select("user_query, bot_response").execute()
        rows = result.data or []
        for row in rows:
            add_chat_to_faiss(row["user_query"], row["bot_response"])
        print(f"✅ Synced {len(rows)} chats from Supabase to FAISS.")
    except Exception as e:
        print("⚠️ Error syncing Supabase history:", e)

# ============================================
# Main Answer Function
# ============================================
def generate_answer(query):
    query = query.strip()
    if not query:
        return "Please enter a message."

    # Greeting Detection
    if is_greeting(query):
        return (
            "👋 Hello! I’m SwarAI.\n\n"
            "I can help you understand and solve queries related to WASAPI, IAudioClient, APOs, "
            "audio processing, and other Windows audio architecture topics.\n\n"
            "Ask me anything related to audio!"
        )

    # Check Supabase for previous answer
    prev_response = get_previous_response(query)
    if prev_response:
        return prev_response

    # Retrieve context from FAISS
    results = get_similar_chunks(query, top_k=5)
    context_text = "\n\n".join([c for c, _ in results]) if results else ""
    if len(context_text) > 4000:
        context_text = context_text[:4000]

    # Build prompt
    prompt_template = """You are a helpful and knowledgeable assistant with deep expertise in Windows Audio Architecture (WASAPI, IAudioClient, APOs, etc.).
Answer the user's question concisely and accurately. Use the context below only if it is relevant.

Context:
{context}

Question:
{user_input}

Answer:"""

    prompt = PromptTemplate(input_variables=["context", "user_input"], template=prompt_template)
    chain = prompt | llm | StrOutputParser()

    # Generate response
    try:
        response_text = chain.invoke({"context": context_text, "user_input": query})
    except Exception as e:
        print("⚠️ Error during response:", e)
        response_text = "Sorry, something went wrong while generating the response."

    # Store in Supabase + FAISS
    try:
        supabase.table("chat_history").insert({
            "user_query": query,
            "bot_response": response_text
        }).execute()
        add_chat_to_faiss(query, response_text)
    except Exception as e:
        print("⚠️ Failed to store chat in Supabase or FAISS:", e)

    return response_text

# ============================================
# Optional one-time sync
# ============================================
if __name__ == "__main__":
    # Run once if you already have history in Supabase
    # sync_supabase_history_to_faiss()
    print("✅ Backend ready. You can now call generate_answer(query).")
