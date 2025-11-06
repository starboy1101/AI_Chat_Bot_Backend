import os
import re
import pickle
import faiss
from sentence_transformers import SentenceTransformer
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
# Globals (Lazy initialization)
# ============================================
embed_model = None
llm = None
embedding_cache = {}

# ============================================
# FAISS / Supabase Config
# ============================================
FAISS_DIR = "faiss_db"
INDEX_FILE = os.path.join(FAISS_DIR, "index.faiss")
METADATA_FILE = os.path.join(FAISS_DIR, "metadata.pkl")

SUPABASE_URL = "https://ghvwpalyqvovyxajdgfb.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImdodndwYWx5cXZvdnl4YWpkZ2ZiIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc2MDUzNDA5MSwiZXhwIjoyMDc2MTEwMDkxfQ.jE1dZRKvZT36EF3hCQ9N2-5viromWqbJP_1vnKSwU7Q"
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ============================================
# Model Loading (Lazy)
# ============================================
def load_models_if_needed():
    """Load models only once to avoid blocking startup."""
    global embed_model, llm
    if embed_model is not None and llm is not None:
        return  # already loaded

    print("🔄 Loading models (lazy init)...")
    try:
        embed_model = SentenceTransformer("all-MiniLM-L6-v2")

        model_path = hf_hub_download(
            repo_id="TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
            filename="tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
        )

        llm = Llama(model_path=model_path, n_ctx=1024, n_batch=64)
        print("✅ Models loaded successfully.")
    except Exception as e:
        print("⚠️ Model loading failed:", e)
        raise

# ============================================
# Utility Functions
# ============================================
def get_embedding(text):
    """Return cached or new embedding for given text."""
    load_models_if_needed()
    if text not in embedding_cache:
        embedding_cache[text] = embed_model.encode([text]).astype("float32")
    return embedding_cache[text]

def get_similar_chunks(query, top_k=5):
    """Fetch top-k most similar chunks from FAISS DB."""
    if not os.path.exists(INDEX_FILE) or not os.path.exists(METADATA_FILE):
        return []
    try:
        index = faiss.read_index(INDEX_FILE)
        with open(METADATA_FILE, "rb") as f:
            metadata = pickle.load(f)
        chunks = metadata.get("chunks", [])
        sources = metadata.get("sources", [])
        q_emb = get_embedding(query)
        D, I = index.search(q_emb, min(top_k, len(chunks)))
        return [(chunks[i], sources[i]) for i in I[0]]
    except Exception as e:
        print("⚠️ FAISS retrieval error:", e)
        return []

def is_greeting(text: str) -> bool:
    greetings = ["hi", "hello", "hey", "hii", "hlo", "yo", "hola"]
    text = text.lower().strip()
    return any(re.search(rf"\b{greet}\b", text) for greet in greetings)

# ============================================
# Main Chat Function
# ============================================
def generate_answer(query):
    """Generate an answer using the model and FAISS context."""
    load_models_if_needed()

    query = query.strip()
    if not query:
        return "Please enter a message."

    if is_greeting(query):
        return (
            "👋 Hello! I’m SwarAI.\n\n"
            "I can help you understand and solve queries related to WASAPI, IAudioClient, APOs, "
            "audio processing, and other Windows audio architecture topics.\n\n"
            "Ask me anything related to audio!"
        )

    # Retrieve context from FAISS
    results = get_similar_chunks(query, top_k=5)
    context_text = "\n\n".join([c for c, _ in results]) if results else ""
    if len(context_text) > 4000:
        context_text = context_text[:4000]

    # Prompt Template (same as your logic)
    prompt_template = """You are a helpful and knowledgeable assistant with deep expertise in Windows Audio Architecture.
Answer the user's question concisely and accurately. Use the context below only if relevant.

Context:
{context}

Question:
{user_input}

Answer:"""

    prompt = PromptTemplate(input_variables=["context", "user_input"], template=prompt_template)

    try:
        response = llm(
            prompt=prompt.format(context=context_text, user_input=query),
            max_tokens=256,
        )
        response_text = response["choices"][0]["text"].strip()
    except Exception as e:
        print("⚠️ Generation error:", e)
        response_text = "Sorry, something went wrong during generation."

    return response_text
