#!/bin/bash

# Setup script for RAG-Based Domain-Specific Q&A System

echo "🚀 Setting up RAG-Based Q&A System..."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 is not installed. Please install Python 3.8+ first."
    exit 1
fi

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📚 Installing Python dependencies..."
pip install -r requirements.txt

echo "🔧 Setup complete! Now you need to:"
echo "1. Install Ollama from https://ollama.ai/"
echo "2. Pull the required models:"
echo "   ollama pull deepseek-r1:1.5b"
echo "   ollama pull llama3.2:1b"
echo "3. Start the application:"
echo "   streamlit run rag_deep.py"

echo "✅ Setup finished successfully!"
