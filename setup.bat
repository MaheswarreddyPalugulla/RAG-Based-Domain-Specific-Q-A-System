@echo off
REM Setup script for RAG-Based Domain-Specific Q&A System (Windows)

echo 🚀 Setting up RAG-Based Q&A System...

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python is not installed. Please install Python 3.8+ first.
    pause
    exit /b 1
)

REM Create virtual environment
echo 📦 Creating virtual environment...
python -m venv venv

REM Activate virtual environment
echo 🔧 Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo ⬆️ Upgrading pip...
python -m pip install --upgrade pip

REM Install requirements
echo 📚 Installing Python dependencies...
pip install -r requirements.txt

echo 🔧 Setup complete! Now you need to:
echo 1. Install Ollama from https://ollama.ai/
echo 2. Pull the required models:
echo    ollama pull deepseek-r1:1.5b
echo    ollama pull llama3.2:1b
echo 3. Start the application:
echo    streamlit run rag_deep.py

echo ✅ Setup finished successfully!
pause
