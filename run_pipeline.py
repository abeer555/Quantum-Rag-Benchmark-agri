#!/usr/bin/env python3
"""
Master Pipeline Runner
Runs web crawler, quantum RAG, classical RAG, and comparison
"""

import subprocess
import sys
from pathlib import Path
import time

def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*80}")
    print(f"🚀 {description}")
    print(f"{'='*80}")
    print(f"Command: {cmd}\n")
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, text=True)
        print(f"\n✅ {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ {description} failed with error code {e.returncode}")
        return False
    except Exception as e:
        print(f"\n❌ {description} failed: {e}")
        return False

def check_dependencies():
    """Check if required packages are installed."""
    print("🔍 Checking dependencies...")
    
    required_packages = [
        'qdrant_client',
        'sentence_transformers',
        'transformers',
        'crawl4ai',
        'pennylane',
        'numpy',
        'tqdm'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print("\nInstall with: pip install -r requirements.txt")
        return False
    
    print("✅ All dependencies installed!\n")
    return True

def main():
    """Run the complete pipeline."""
    print("""
╔════════════════════════════════════════════════════════════════════════════╗
║                  QUANTUM RAG BENCHMARK PIPELINE                            ║
║                  Agricultural Data Collection & Analysis                    ║
╚════════════════════════════════════════════════════════════════════════════╝
""")
    
    # Check dependencies
    if not check_dependencies():
        print("\n❌ Please install dependencies first")
        sys.exit(1)
    
    # Check for data
    data_dir = Path("agricultural_data_complete/txt")
    
    if not data_dir.exists() or not list(data_dir.glob("*.txt")):
        print("\n📥 No data found. Running web crawler...")
        print("⚠️  This will take 3-5 minutes with polite delays")
        
        response = input("\nRun web crawler now? (y/n): ").strip().lower()
        if response == 'y':
            if not run_command("python web_crawler.py", "Web Crawler Data Collection"):
                print("\n❌ Pipeline failed at web crawler stage")
                sys.exit(1)
        else:
            print("\n⚠️  Skipping web crawler. You can run it manually:")
            print("   python web_crawler.py")
            print("\nContinuing with existing data (if any)...")
    else:
        print(f"\n✅ Data found in {data_dir}")
        num_files = len(list(data_dir.glob("*.txt")))
        print(f"   {num_files} TXT files ready for processing")
    
    # Recheck data
    if not data_dir.exists() or not list(data_dir.glob("*.txt")):
        print("\n❌ No data available. Cannot proceed.")
        sys.exit(1)
    
    print("\n" + "="*80)
    print("📊 RUNNING COMPARISON")
    print("="*80)
    
    # Run comparison
    if not run_command("python compare_rag_results.py", "RAG Systems Comparison"):
        print("\n⚠️  Comparison failed, but you can still run systems manually:")
        print("   - Classical RAG: python classical_rag.py")
        print("   - Quantum RAG: python quantum_rag.py")
    else:
        print("\n" + "="*80)
        print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*80)
        print("\n📊 Results saved to:")
        print("   - rag_comparison_results.json")
        print("   - rag_comparison_results.csv")
        print("\n💡 You can also run the systems interactively:")
        print("   - Classical RAG: python classical_rag.py")
        print("   - Quantum RAG: python quantum_rag.py")
    
    print("\n✅ Done!")

if __name__ == "__main__":
    main()
