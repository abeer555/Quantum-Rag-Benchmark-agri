# Quantum RAG Benchmark - Agriculture

Comprehensive benchmark comparing Quantum-Enhanced RAG vs Classical RAG for agricultural information retrieval using real-world data from 50+ trusted sources.

## Project Structure

```
Quantum-Rag-Benchmark-agri/
├── web_crawler.py              # Scrapes 50+ agricultural sources
├── quantum_rag.py              # Quantum-enhanced RAG system
├── classical_rag.py            # Classical RAG baseline
├── compare_rag_results.py      # Automated comparison script
├── run_pipeline.py             # Master pipeline runner
├── requirements.txt            # Python dependencies
├── setup_crawler.sh            # Crawler setup script
│
├── agricultural_data_complete/ # Web crawler output
│   ├── txt/                    # Clean TXT files (50+ sources)
│   ├── json/                   # Metadata files
│   └── logs/                   # Scraping statistics
│
├── old/                        # Original implementation
│   └── src/
│       ├── quantum_embeddings/ # Quantum feature maps
│       ├── quantum_rag.py      # Original quantum RAG
│       └── baseline_rag.py     # Original baseline
│
└── *.csv, *.json               # Results and logs
```

## Features

### 🌐 Web Crawler

- **50+ Trusted Sources**: FAO, USDA, World Bank, CGIAR, research institutions
- **Clean TXT Output**: Perfect for RAG systems
- **Organized Storage**: Separate folders for txt, json, logs
- **Automatic Merging**: Creates unified corpus file

### ⚛️ Quantum RAG

- **Quantum Feature Maps**: Angle, Amplitude, IQP embeddings
- **PennyLane & Qiskit**: Multiple quantum backends
- **Hybrid Embeddings**: Combines classical + quantum features
- **Configurable Qubits**: 4-16 qubits supported

### 📊 Classical RAG (Baseline)

- **MiniLM Embeddings**: Fast, 384-dimensional
- **Qdrant Vector DB**: In-memory for speed
- **T5 Generation**: Local answer generation
- **Gemini API Support**: Optional cloud LLM

### 🔬 Comparison Framework

- **Automated Benchmarking**: 10 test queries
- **Multiple Metrics**: Speed, similarity, diversity, overlap
- **Statistical Analysis**: Aggregate results
- **Export Formats**: JSON and CSV

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
playwright install  # For web crawler
```

### 2. Run Complete Pipeline

```bash
python run_pipeline.py
```

This will:

1. Check dependencies
2. Run web crawler (if no data exists)
3. Run automated comparison
4. Generate results

### 3. Or Run Components Individually

```bash
# Step 1: Collect data (3-5 minutes)
python web_crawler.py

# Step 2: Run comparison
python compare_rag_results.py

# Step 3: Try interactive systems
python quantum_rag.py      # Quantum-enhanced
python classical_rag.py    # Classical baseline
```

## Usage Examples

### Web Crawler

```bash
# Full scraping (50+ sources)
python web_crawler.py

# Quick sample (10 sources for testing)
# Edit web_crawler.py and uncomment:
# asyncio.run(scrape_quick_sample())
```

### Quantum RAG

```bash
python quantum_rag.py
```

Options:

- **Embedding Type**: angle, amplitude, classical
- **Qubits**: 4-16 (default: 8)
- **Interactive Mode**: Ask questions in real-time

### Classical RAG

```bash
python classical_rag.py
```

Features:

- Fast classical embeddings
- Same interface as Quantum RAG
- Logs to CSV for comparison

### Comparison

```bash
python compare_rag_results.py
```

Outputs:

- `rag_comparison_results.json` - Detailed results
- `rag_comparison_results.csv` - Spreadsheet format

## Results & Metrics

The comparison evaluates:

1. **Retrieval Speed**

   - Average time per query
   - Speed ratio (quantum vs classical)

2. **Retrieval Quality**

   - Average similarity scores
   - Top-k overlap between systems

3. **Source Diversity**

   - Variety of sources in results
   - Coverage across corpus

4. **Statistical Significance**
   - Aggregate metrics
   - Per-query analysis

## Configuration

### Quantum Settings

In `quantum_rag.py`:

```python
# Embedding types
- angle: Simple rotation-based (fast)
- amplitude: Dense state preparation (expressive)
- iqp: Instantaneous Quantum Polynomial (complex)

# Qubits
n_qubits = 8  # 4-16 recommended
```

### Classical Settings

In `classical_rag.py`:

```python
# Embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Chunk settings
chunk_size = 500  # words
overlap = 50      # words
```

### Crawler Settings

In `web_crawler.py`:

```python
# Delay between requests (be polite!)
delay = 3  # seconds

# Retry attempts
max_retries = 3
```

## Data Sources

The web crawler collects from:

**International Organizations (10)**

- FAO (Food and Agriculture Organization)
- World Bank Agriculture
- CGIAR Research

**US Government (8)**

- USDA Farming, Crops, Livestock
- Economic Research Service
- National Agricultural Statistics

**Other Governments (5)**

- UK DEFRA
- EU Agriculture
- Australia, India

**Research Institutions (10+)**

- CIMMYT, IRRI, ICRISAT
- IFPRI, CSIRO

**Other (7)**

- AGRIS Database
- Precision Agriculture
- Sustainable Agriculture

## Requirements

### Python Packages

- qdrant-client >= 1.7.0
- sentence-transformers >= 2.2.2
- transformers >= 4.35.0
- crawl4ai >= 0.2.0
- pennylane >= 0.33.0
- qiskit >= 0.45.0
- numpy, pandas, tqdm

### Optional

- google-generativeai (for Gemini API)
- python-dotenv (for .env support)

### Hardware

- CPU: Any modern processor
- RAM: 8GB+ recommended
- Storage: 1GB for data
- GPU: Not required (CPU mode)

## Gemini API (Optional)

For better answer generation:

1. Get API key from https://makersuite.google.com/app/apikey

2. Create `.env` file:

```bash
GEMINI_API_KEY=your-api-key-here
```

3. Run any RAG system - it will auto-detect Gemini

## Troubleshooting

### Import Errors

```bash
pip install -r requirements.txt
playwright install
```

### No Data Found

```bash
python web_crawler.py
# Wait 3-5 minutes for completion
```

### Quantum Import Fails

```bash
pip install pennylane qiskit
# Or use classical mode
```

### Crawler Fails

- Check internet connection
- Some sites may block automated access
- URLs may have changed
- Check logs in `agricultural_data_complete/logs/`

## Performance Tips

1. **Use Quick Sample First**: Test with 10 sources before full scraping
2. **Classical for Speed**: Use classical RAG for fast prototyping
3. **Lower Qubits**: Start with 4-6 qubits for faster quantum processing
4. **Cache Models**: Models are cached after first download

## Project Timeline

1. **Data Collection**: 3-5 minutes (web crawler)
2. **Indexing**: 2-3 minutes (first time)
3. **Comparison**: 1-2 minutes (10 queries)
4. **Interactive Use**: Real-time queries

Total: ~10 minutes for complete pipeline

## Results Format

### JSON Output

```json
{
  "timestamp": "2025-11-03T...",
  "num_queries": 10,
  "queries": [...],
  "aggregates": {
    "classical": {...},
    "quantum": {...},
    "comparison": {...}
  }
}
```

### CSV Output

Columns: query, classical_time, classical_similarity, quantum_time, quantum_similarity, overlap, speedup, score_improvement_pct

## Contributing

Contributions welcome! Areas:

- Additional data sources
- New quantum feature maps
- Evaluation metrics
- Visualization tools

## License

See LICENSE file in repository

## Citation

If you use this benchmark in research:

```
@software{quantum_rag_agri_2025,
  title={Quantum RAG Benchmark - Agriculture},
  author={...},
  year={2025},
  url={https://github.com/abeer555/Quantum-Rag-Benchmark-agri}
}
```

## Support

- Issues: GitHub Issues
- Docs: See WEB_CRAWLER_README.md for crawler details
- Contact: Repository owner

---

**Built with**: PennyLane, Qiskit, Qdrant, Sentence Transformers, Crawl4AI
