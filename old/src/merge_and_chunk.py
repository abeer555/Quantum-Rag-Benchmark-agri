from pathlib import Path
import json
import re

INPUT_FILES = [
    "data/raw/agricultural_corpus.txt",           # FAO Yearbook
    "data/raw/wikipedia_agriculture_expanded.txt" # Expanded Wikipedia
]
OUTPUT_FILE = Path("data/cleaned.jsonl")

CHUNK_SIZE = 400
OVERLAP = 50

def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def chunk_text(text: str, chunk_size=400, overlap=50):
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks

def main():
    merged_text = ""
    for file in INPUT_FILES:
        if Path(file).exists():
            with open(file, "r", encoding="utf-8") as f:
                merged_text += f.read() + "\n\n"
        else:
            print(f"⚠️ Missing file: {file}")

    print(f"📚 Total raw words: {len(merged_text.split()):,}")

    chunks = chunk_text(clean_text(merged_text), CHUNK_SIZE, OVERLAP)
    print(f"✂️ Created {len(chunks)} chunks")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        for i, chunk in enumerate(chunks):
            out.write(json.dumps({"id": f"chunk_{i}", "text": chunk}) + "\n")

    print(f"✅ Saved cleaned corpus → {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
