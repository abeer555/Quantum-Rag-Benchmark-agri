#!/bin/bash
# Run expanded web crawler to collect more agricultural data
# This will give us more data points for better complexity analysis

echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║           EXPANDED AGRICULTURAL DATA COLLECTION                            ║"
echo "║     Crawling 100+ sources for comprehensive dataset analysis               ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo ""

# Activate virtual environment if exists
if [ -f "/home/abeer/.config/pythonVirtualEnv/venv/bin/activate" ]; then
    source /home/abeer/.config/pythonVirtualEnv/venv/bin/activate
    echo "✅ Virtual environment activated"
else
    echo "⚠️  No virtual environment found, using system Python"
fi

echo ""
echo "📊 Starting web crawler..."
echo "   Target: 100+ agricultural sources"
echo "   Output: agricultural_data_complete/txt/"
echo ""

# Run the web crawler
python web_crawler.py

echo ""
echo "✅ Data collection complete!"
echo ""
echo "📈 Now you can run: python advanced_rag_comparison.py"
echo "   This will test with 12 dataset sizes: [5, 10, 15, 20, 25, 30, 40, 50, 60, 75, 90, 110]"
echo "   More data points = clearer O(log n) vs O(n) patterns!"
