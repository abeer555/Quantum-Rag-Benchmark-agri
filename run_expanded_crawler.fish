#!/usr/bin/env fish
# Fish shell version of the expanded crawler runner

echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║           EXPANDED AGRICULTURAL DATA COLLECTION                            ║"
echo "║     Crawling 100+ sources for comprehensive dataset analysis               ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo ""

# Activate virtual environment if exists
if test -f "/home/abeer/.config/pythonVirtualEnv/venv/bin/activate.fish"
    source /home/abeer/.config/pythonVirtualEnv/venv/bin/activate.fish
    echo "✅ Virtual environment activated"
else
    echo "⚠️  No virtual environment found, using system Python"
end

echo ""
echo "📊 Current dataset status:"
set current_files (count agricultural_data_complete/txt/*.txt 2>/dev/null)
echo "   Current files: $current_files"
echo "   Target: 100+ files"
echo ""

echo "📡 Starting web crawler..."
echo "   This will take several minutes..."
echo ""

# Run the web crawler
python web_crawler.py

echo ""
echo "✅ Data collection complete!"
echo ""

set final_files (count agricultural_data_complete/txt/*.txt 2>/dev/null)
echo "📊 Final dataset: $final_files files"
echo ""
echo "📈 Next step: python advanced_rag_comparison.py"
echo "   Will test with 12 dataset sizes for thorough analysis"
echo "   Expected runtime: 10-15 minutes"
