import json
from pathlib import Path

# Input and output paths
INPUT_FILE = Path("agricultural_rag_corpus/agricultural_corpus.txt")
OUTPUT_FILE = Path("data/cleaned.jsonl")

# Parameters
CHUNK_SIZE = 400    # words per chunk
OVERLAP = 50        # overlapping words between chunks

def chunk_text(text, chunk_size=400, overlap=50):
    """Split text into overlapping word chunks"""
    words = text.split()
    chunks, i = [], 0
    
    while i < len(words):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    
    return chunks

def main():
    # Make sure output folder exists
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    # Read input file
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        text = f.read()
    
    # Split into chunks
    chunks = chunk_text(text, CHUNK_SIZE, OVERLAP)
    print(f"✅ Created {len(chunks)} chunks from {INPUT_FILE}")
    
    # Save to JSONL
    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        for i, chunk in enumerate(chunks):
            record = {
                "id": f"corpus_{i}",
                "text": chunk
            }
            out.write(json.dumps(record) + "\n")
    
    print(f"💾 Saved chunks to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
