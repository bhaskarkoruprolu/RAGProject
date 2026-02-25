from flask import Flask, render_template, request, jsonify
from src.search import RAGSearch
import os

app = Flask(__name__)

# Initialize RAG Engine
# We initialize it globally so it loads the vector store only once on startup
rag_engine = RAGSearch()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    query = data.get('query')
    
    if not query:
        return jsonify({"error": "No query provided"}), 400
    
    try:
        # Perform search and summarization
        # You can adjust top_k here or make it a parameter
        summary = rag_engine.search_and_summarize(query, top_k=3)
        return jsonify({"summary": summary})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Use PORT env variable for deployment, default to 5000
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(host='0.0.0.0', port=port, debug=debug)
