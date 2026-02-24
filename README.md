# RAG Assistant 🤖

A powerful Retrieval-Augmented Generation (RAG) application that enables you to chat with your local documents using LangChain, OpenAI, and FAISS.

## 🚀 Features
- **Multi-Format Support**: Load PDFs, TXT, CSV, Excel (XLSX), Word (DOCX), and JSON files.
- **Fast Vector Search**: Uses FAISS for efficient similarity search across document chunks.
- **Modern UI**: Clean web interface built with Flask for interactive chatting.
- **Intelligent Summarization**: Leverages OpenAI's `gpt-4o-mini` to provide concise answers based on document context.
- **Docker Ready**: Includes a multi-stage Dockerfile for easy containerized deployment.

## 🛠️ Project Structure
```text
RAGProject/
├── src/                # Core RAG logic (loading, embeddings, search)
├── data/               # Your documents location
├── templates/          # HTML files for the web UI
├── server.py           # Flask web server entry point
├── app.py              # CLI entry point for testing
├── Dockerfile          # Container configuration
└── Requirements.txt    # Python dependencies
```

## ⚙️ Setup Instructions

### 1. Prerequisites
- Python 3.10+ (Tested on 3.13)
- OpenAI API Key

### 2. Installation
Clone the repository and install dependencies:
```powershell
# Activate your virtual environment
.\venv\Scripts\activate

# Install requirements
pip install -r Requirements.txt
```

### 3. Environment Configuration
Create a `.env` file in the root directory and add your OpenAI API key:
```env
OPENAI_API_KEY=your_api_key_here
```

### 4. Running the Application
To start the web interface:
```powershell
python server.py
```
Visit `http://127.0.0.1:5000` in your browser.

## 🐳 Docker Deployment
To build and run using Docker:
```bash
docker build -t rag-assistant .
docker run -p 5000:5000 --env-file .env rag-assistant
```

## 📝 License
This project is for educational and personal use.
