# Enhanced Agricultural Web Crawler

## Overview

This comprehensive web crawler scrapes agricultural data from 50+ trusted sources worldwide and converts all content to clean TXT format suitable for RAG (Retrieval Augmented Generation) systems.

## Features

### ✅ Comprehensive Data Sources

- **International Organizations**: FAO, World Bank, CGIAR
- **Government Agencies**: USDA, DEFRA, EU Agriculture, Australia, India
- **Research Institutions**: CIMMYT, IRRI, ICRISAT, IFPRI, CSIRO
- **Topics Covered**: Crops, Livestock, Climate, Soil, Water, Technology, Sustainability

### ✅ Clean Text Output

- Converts markdown to plain text
- Removes URLs, special characters, and formatting
- Creates structured TXT files with headers
- Merges all sources into single corpus file

### ✅ Organized Storage

```
agricultural_data_complete/
├── txt/              # Clean TXT files (one per source)
├── json/             # Metadata for each source
└── logs/             # Scraping statistics and reports
```

### ✅ Robust Scraping

- Retry mechanism (3 attempts per source)
- Polite delays between requests
- Error handling and logging
- Progress tracking

## Installation

```bash
# Install the crawl4ai library
pip install crawl4ai

# Optionally, install browser drivers if needed
playwright install
```

## Usage

### 1. Full Comprehensive Scraping (50+ sources)

```bash
python web_crawler.py
```

This will scrape all sources and create:

- Individual TXT files for each source
- A merged corpus file: `agricultural_corpus_complete.txt`
- Detailed statistics and logs

**Estimated time**: 3-5 minutes (with 3-second delays)
**Output size**: 10-50 MB depending on source availability

### 2. Quick Sample (10 sources - for testing)

Edit `web_crawler.py` and uncomment:

```python
asyncio.run(scrape_quick_sample())
```

Then run:

```bash
python web_crawler.py
```

### 3. Topic-Specific Scraping (30+ topic sources)

Edit `web_crawler.py` and uncomment:

```python
asyncio.run(scrape_specific_topics())
```

Topics include:

- Crop Production
- Livestock Management
- Climate Smart Agriculture
- Soil & Water Management
- Agricultural Technology
- Food Security
- Sustainable Agriculture
- And more...

### 4. Research Institutions Only (15+ sources)

Edit `web_crawler.py` and uncomment:

```python
asyncio.run(scrape_research_institutions())
```

## Data Sources

### International Organizations (10 sources)

- FAO (Food and Agriculture Organization) - Multiple sections
- World Bank Agriculture
- CGIAR Research

### US Government (8 sources)

- USDA Topics (Farming, Crops, Livestock, Research, Conservation, Organic)
- USDA Economic Research Service
- USDA National Agricultural Statistics Service

### Other Governments (5 sources)

- UK DEFRA
- EU Common Agricultural Policy
- Australia Department of Agriculture
- CSIRO Agriculture
- ICAR India

### Research Institutions (10+ sources)

- CGIAR Centers (CIMMYT, IRRI, ICRISAT, CIP, IITA, ICARDA, WorldFish)
- IFPRI (International Food Policy Research Institute)
- CABI
- National research centers

### Other Sources (7 sources)

- AGRIS (FAO Agricultural Research Database)
- Precision Agriculture
- Sustainable Agriculture Network
- AgFunder News

## Output Format

### Individual TXT Files

Each source creates a TXT file with this format:

```
================================================================================
SOURCE: fao_crops
URL: https://www.fao.org/crop-production/en/
SCRAPED: 2025-11-02 14:30:45
CONTENT LENGTH: 45000 characters
================================================================================

[Clean plain text content...]
```

### Merged Corpus File

All individual files are merged into one large corpus:

- `agricultural_corpus_complete.txt` (Full scraping)
- `agricultural_topics_corpus.txt` (Topics only)
- `agricultural_research_corpus.txt` (Research institutions)
- `agricultural_sample_corpus.txt` (Quick sample)

### Metadata Files (JSON)

Each source has a metadata file with:

```json
{
  "source_name": "fao_crops",
  "url": "https://www.fao.org/crop-production/en/",
  "scraped_at": "2025-11-02T14:30:45",
  "content_length": 45000,
  "success": true,
  "links_count": 150,
  "txt_file": "/path/to/txt/fao_crops.txt"
}
```

### Statistics Report

After scraping, a summary is generated:

```
================================================================================
AGRICULTURAL DATA SCRAPING SUMMARY
================================================================================
Total Sources Attempted: 52
Successful: 48
Failed: 4
Success Rate: 92.3%
Total Content: 25,450,000 characters (24.28 MB)
Average per Source: 530,208 characters
================================================================================
```

## Customization

### Add More Sources

Edit `AGRICULTURAL_SOURCES` dictionary in `web_crawler.py`:

```python
AGRICULTURAL_SOURCES = {
    "your_source_name": "https://your-url.com",
    # ... more sources
}
```

### Adjust Scraping Parameters

In the `scrape_to_txt` method:

```python
# Increase timeout for slow sites
page_timeout=60000,  # 60 seconds

# Adjust retry attempts
max_retries=3

# Change delay between requests (be polite!)
await scraper.scrape_all_sources(AGRICULTURAL_SOURCES, delay=3)
```

### Text Cleaning

Modify the `clean_text()` and `markdown_to_txt()` methods to customize text processing.

## Best Practices

1. **Be Respectful**: The crawler includes delays between requests (3 seconds default). Don't reduce this too much.

2. **Test First**: Use `scrape_quick_sample()` to test before running full scraping.

3. **Check Robots.txt**: Ensure you're allowed to scrape from each source.

4. **Monitor Output**: Watch for failed sources and adjust URLs if needed.

5. **Storage Space**: Full scraping can generate 20-100 MB. Ensure you have space.

## Troubleshooting

### "Import crawl4ai could not be resolved"

```bash
pip install crawl4ai
playwright install
```

### Sources Failing to Scrape

- Some sites may block automated access
- URLs may have changed
- Network issues or timeouts
- Check the logs in `logs/` directory

### Low Content Quality

- Some pages may be mostly navigation/menus
- Adjust the `clean_text()` method
- Use topic-specific scraping for better targeting

### Memory Issues

- Scrape in batches (use topic-specific or research-only modes)
- Reduce the number of sources
- Don't merge files immediately

## Integration with RAG System

After scraping, use the TXT files in your RAG pipeline:

```python
# Use the merged corpus
corpus_file = "agricultural_data_complete/agricultural_corpus_complete.txt"

# Or use individual files
txt_files = list(Path("agricultural_data_complete/txt").glob("*.txt"))

# Process for your RAG system
for txt_file in txt_files:
    with open(txt_file, 'r') as f:
        content = f.read()
        # Your chunking and embedding code here
```

## Future Enhancements

Potential additions:

- PDF document downloading and parsing
- Multi-language support
- Recursive link following for deeper crawling
- Image caption extraction
- Table data extraction
- Scheduled periodic scraping
- Duplicate content detection

## License

Use responsibly and in accordance with each website's terms of service and robots.txt.

## Support

For issues or questions, check:

- Crawl4AI documentation: https://github.com/unclecode/crawl4ai
- Individual website terms of service
- Your local internet regulations

---

**Note**: This crawler is designed for research and educational purposes. Always respect website policies and rate limits.
