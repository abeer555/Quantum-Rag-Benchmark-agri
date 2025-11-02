#!/bin/bash

# Enhanced RAG with Gemini API Setup Script

echo "🌾 Setting up Enhanced Agricultural RAG with Gemini API"
echo "======================================================"

# Install required packages
echo "📦 Installing required packages..."
pip install google-generativeai qdrant-client sentence-transformers transformers

echo ""
echo "🔑 Gemini API Key Setup:"
echo "1. Get your Gemini API key from: https://makersuite.google.com/app/apikey"
echo "2. Set the environment variable:"
echo "   export GEMINI_API_KEY='your-api-key-here'"
echo ""
echo "For permanent setup, add to your ~/.bashrc or ~/.zshrc:"
echo "   echo 'export GEMINI_API_KEY=\"your-api-key-here\"' >> ~/.bashrc"
echo ""
echo "🚀 Usage:"
echo "   python src/better_rag.py"
echo ""
echo "✅ Setup complete! The RAG system will:"
echo "   • Use Gemini API for high-quality answers (if API key is set)"
echo "   • Fallback to local T5 model if Gemini is unavailable"
echo "   • Use extractive answers as final fallback"