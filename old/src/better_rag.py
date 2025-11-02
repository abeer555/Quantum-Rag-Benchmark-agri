import json
import time
import csv
import os
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from tqdm import tqdm

# Environment file loading
try:
    from dotenv import load_dotenv
    HAS_DOTENV = True
except ImportError:
    HAS_DOTENV = False

# Gemini API integration
try:
    from google import genai
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False
    print("⚠️  Gemini API not available. Install with: pip install google-generativeai")

# Paths
DATA_FILE = Path("data/cleaned.jsonl")
COLLECTION_NAME = "agri_rag"
LOG_FILE = Path("data/query_logs.csv")

# -------------------- INIT --------------------
def init_pipeline():
    print("🔄 Loading embedding model (MiniLM, fast on CPU)...")
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    print("🔄 Connecting to Qdrant (in-memory)...")
    client = QdrantClient(":memory:")

    print("🔄 Loading local text-generation model (flan-t5-small for memory efficiency)...")
    # Use smaller model and force CPU to avoid CUDA memory issues
    generator = pipeline(
        "text2text-generation", 
        model="google/flan-t5-large",
        device=-1  # Force CPU usage
    )
    
    # Initialize Gemini API
    gemini_config = init_gemini()

    return embedder, client, generator, gemini_config

def init_gemini():
    """Initialize Gemini API if available and configured."""
    if not HAS_GEMINI:
        print("⚠️  Gemini API not available")
        return None
    
    # Load .env file if available
    if HAS_DOTENV:
        env_file = Path(".env")
        if env_file.exists():
            load_dotenv(env_file)
            print("✅ Loaded environment variables from .env file")
        else:
            load_dotenv()  # Try to load from default locations
    
    # Check for API key in environment variable
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("⚠️  GEMINI_API_KEY not found")
        print("   Options:")
        print("   1. Create .env file with: GEMINI_API_KEY=your-api-key")
        print("   2. Set environment variable: export GEMINI_API_KEY='your-api-key'")
        return None
    
    # Set the API key for Google AI client (it expects GOOGLE_API_KEY)
    os.environ['GOOGLE_API_KEY'] = api_key

    try:
        # Use the exact format from documentation
        client = genai.Client()
        
        # Try different model names, prioritizing pro models first
        model_names = [
            'gemini-1.5-pro',        # Good performance, prioritized
            'gemini-2.5-pro',        # Latest pro model  
            'gemini-1.5-flash',      # Fast and usually available
            'gemini-2.0-flash-exp'   # Experimental flash model
        ]
        
        for model_name in model_names:
            try:
                # Test the model with a simple query using exact documentation format
                prompt = "Test"
                test_response = client.models.generate_content(
                    model=model_name,
                    contents=prompt
                )
                if test_response and test_response.text:
                    print(f"✅ Gemini API initialized successfully with model: {model_name}")
                    return {'client': client, 'model': model_name}
            except Exception as e:
                error_msg = str(e)
                if "503" in error_msg or "overloaded" in error_msg.lower():
                    print(f"⚠️  Model {model_name} overloaded, trying next...")
                elif "404" in error_msg or "not found" in error_msg.lower():
                    print(f"⚠️  Model {model_name} not available, trying next...")
                else:
                    print(f"⚠️  Model {model_name} failed, trying next...")
                continue
        
        print("❌ No available Gemini models found (may be temporarily overloaded)")
        return None
        
    except Exception as e:
        print(f"⚠️  Failed to initialize Gemini API: {e}")
        return None

# -------------------- DATA LOADING --------------------
def load_chunks(file_path):
    records = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return records

# -------------------- INDEXING --------------------
def index_chunks(records, embedder, client, batch_size=32):
    dim = embedder.get_sentence_embedding_dimension()

    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )

    print(f"🔎 Embedding {len(records)} chunks in batches of {batch_size}...")
    texts = [r["text"] for r in records]
    vectors = embedder.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True
    )

    points = [
        PointStruct(id=i, vector=vec.tolist(), payload=records[i])
        for i, vec in enumerate(vectors)
    ]

    client.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"✅ Indexed {len(records)} chunks into {COLLECTION_NAME}")

# -------------------- SEARCH --------------------
def search(query, embedder, client, top_k=5):
    t0 = time.time()
    qvec = embedder.encode(query).tolist()
    t1 = time.time()

    hits = client.query_points(
        collection_name=COLLECTION_NAME,
        query=qvec,
        limit=top_k
    ).points

    t2 = time.time()

    embedding_time = t1 - t0
    retrieval_time = t2 - t1
    total_time = t2 - t0

    contexts = [h.payload.get("text", "") for h in hits]
    return contexts, embedding_time, retrieval_time, total_time

# -------------------- GENERATION --------------------
def generate_answer(query, contexts, generator, gemini_config=None, words_per_chunk=60, max_chunks=3):
    """
    Generate answer using Gemini API as primary, with T5 and extractive as fallbacks.
    """
    if not contexts:
        return "I don't have enough information to answer that question."
    
    # Try Gemini first for best quality
    if gemini_config:
        print("🔮 Generating answer with Gemini API...")
        gemini_answer = generate_gemini_answer(query, contexts, gemini_config)
        if gemini_answer:
            return f"{gemini_answer} [Generated by Gemini]"
    
    # Fallback to T5 model
    print("🤖 Falling back to local T5 model...")
    
    # Build focused context from retrieved documents
    compact_contexts = []
    for i, ctx in enumerate(contexts[:max_chunks]):
        words = ctx.split()[:words_per_chunk]
        compact_context = " ".join(words)
        compact_contexts.append(compact_context)

    context = " ".join(compact_contexts)

    # Simplified prompt that works better with T5
    prompt = f"Question: {query}\nContext: {context}\nAnswer:"

    try:
        result = generator(
            prompt, 
            max_new_tokens=100, 
            do_sample=True,
            temperature=0.5,
            pad_token_id=generator.tokenizer.eos_token_id,
            repetition_penalty=1.2
        )
        
        # Extract the generated text
        generated = result[0]["generated_text"]
        
        # For T5, remove the input prompt from the output if it's included
        if prompt in generated:
            answer = generated.replace(prompt, "").strip()
        else:
            answer = generated.strip()
        
        # Clean up the answer
        if answer:
            # Remove incomplete sentences at the end
            sentences = answer.split('.')
            if len(sentences) > 1 and sentences[-1].strip() and len(sentences[-1].strip()) < 10:
                answer = '.'.join(sentences[:-1]) + '.'
            
            # Ensure it's not too short or repetitive
            if len(answer.split()) < 5:
                return f"Based on the available information: {contexts[0][:100]}..."
            
            return f"{answer} [Generated by T5]"
        else:
            return f"Based on the agricultural sources: {contexts[0][:100]}..."
        
    except Exception as e:
        print(f"T5 Generation error: {e}")
        # Final fallback to extractive answer
        print("📝 Using extractive fallback...")
        return f"{create_extractive_answer(query, contexts)} [Extractive answer]"


def generate_gemini_answer(query, contexts, gemini_config):
    """
    Generate answer using Gemini API for high quality responses.
    """
    if not gemini_config:
        return None
    
    client = gemini_config['client']
    model_name = gemini_config['model']
    
    try:
        # Prepare context from retrieved documents
        context_text = "\n\n".join([f"Source {i+1}: {ctx[:400]}" for i, ctx in enumerate(contexts[:3])])
        
        # Create comprehensive prompt for Gemini
        prompt = f"""You are an expert agricultural advisor. Answer the following question based on the provided context information.

Question: {query}

Context Information:
{context_text}

Instructions:
1. Provide a clear, practical answer based on the context provided
2. Include specific details and recommendations when available
3. If the context doesn't fully address the question, mention this limitation
4. Keep the answer concise but informative (2-4 sentences)
5. Focus on actionable agricultural advice

Answer:"""
        
        # Use exact format from documentation
        response = client.models.generate_content(
            model=model_name,
            contents=prompt
        )
        
        if response and response.text:
            return response.text.strip()
        else:
            return None
            
    except Exception as e:
        print(f"Gemini API error: {e}")
        return None

def create_extractive_answer(query, contexts):
    """
    Create an extractive answer when generative model fails.
    """
    if not contexts:
        return "No relevant information found."
    
    query_lower = query.lower()
    best_context = contexts[0]
    
    # Simple keyword matching for better context selection
    for ctx in contexts:
        ctx_lower = ctx.lower()
        # Count matching keywords
        query_words = set(query_lower.split())
        ctx_words = set(ctx_lower.split())
        overlap = len(query_words.intersection(ctx_words))
        
        if overlap > 0:
            best_context = ctx
            break
    
    # Extract relevant sentences
    sentences = best_context.split('.')
    relevant_sentences = []
    
    for sentence in sentences:
        sentence_lower = sentence.lower()
        if any(word in sentence_lower for word in query_lower.split() if len(word) > 2):
            relevant_sentences.append(sentence.strip())
    
    if relevant_sentences:
        # Take first 2 relevant sentences
        answer = '. '.join(relevant_sentences[:2])
        if not answer.endswith('.'):
            answer += '.'
        return answer
    else:
        # Fallback to first part of best context
        words = best_context.split()[:30]
        answer = ' '.join(words)
        if not answer.endswith('.'):
            answer += '...'
        return answer



# -------------------- LOGGING --------------------
def log_results(query, answer, embedding_time, retrieval_time, total_time):
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    header = ["query", "embedding_time", "retrieval_time", "total_time", "answer"]

    new_file = not LOG_FILE.exists()
    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(header)
        writer.writerow([query, f"{embedding_time:.4f}", f"{retrieval_time:.4f}",
                         f"{total_time:.4f}", answer[:200]])

# -------------------- MAIN --------------------
if __name__ == "__main__":
    records = load_chunks(DATA_FILE)
    embedder, client, generator, gemini_config = init_pipeline()
    index_chunks(records, embedder, client, batch_size=32)

    print("\n🤖 Advanced Agricultural RAG with Gemini API (type 'exit' to quit)")
    if gemini_config:
        print(f"✅ Gemini API enabled for high-quality answers (model: {gemini_config['model']})")
    else:
        print("⚠️  Using local models only (set GEMINI_API_KEY for better answers)")
    print()
    
    while True:
        query = input("❓ Your question: ")
        if query.lower() in ["exit", "quit"]:
            break

        if not query.strip():
            continue
            
        print(f"\n🔍 Processing: '{query}'...")

        contexts, et, rt, tt = search(query, embedder, client, top_k=5)
        
        if not contexts:
            print("❌ No relevant information found.")
            continue
            
        print(f"📚 Found {len(contexts)} relevant sources")
        
        # Generate answer with Gemini integration
        start_gen = time.time()
        answer = generate_answer(query, contexts, generator, gemini_config)
        gen_time = time.time() - start_gen

        print("\n--- Generated Answer ---")
        print(answer)
        
        print(f"\n📖 Top Sources:")
        for i, ctx in enumerate(contexts[:3]):
            print(f"  {i+1}. {ctx[:80]}...")

        total_with_gen = tt + gen_time
        print(f"\n⏱ Embedding: {et:.4f}s | Retrieval: {rt:.4f}s | Generation: {gen_time:.4f}s | Total: {total_with_gen:.4f}s\n")

        log_results(query, answer, et, rt, total_with_gen)
