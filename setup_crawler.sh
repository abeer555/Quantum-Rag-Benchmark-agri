#!/bin/bash

echo "=========================================="
echo "Agricultural Web Crawler Setup"
echo "=========================================="
echo ""

# Check Python version
echo "Checking Python version..."
python3 --version

if [ $? -ne 0 ]; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

echo ""
echo "Installing required packages..."
echo ""

# Install crawl4ai
echo "1. Installing crawl4ai..."
pip install crawl4ai

if [ $? -ne 0 ]; then
    echo "❌ Failed to install crawl4ai"
    exit 1
fi

# Install playwright browsers (needed for crawl4ai)
echo ""
echo "2. Installing Playwright browsers..."
playwright install

if [ $? -ne 0 ]; then
    echo "⚠️  Playwright installation failed, but you can try running the crawler anyway"
fi

echo ""
echo "=========================================="
echo "✅ Setup Complete!"
echo "=========================================="
echo ""
echo "You can now run the crawler:"
echo ""
echo "  Full scraping (50+ sources):"
echo "    python web_crawler.py"
echo ""
echo "  Quick test (10 sources):"
echo "    Edit web_crawler.py and uncomment: asyncio.run(scrape_quick_sample())"
echo ""
echo "For more details, see WEB_CRAWLER_README.md"
echo ""
