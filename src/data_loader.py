from pathlib import Path
from typing import List, Any
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders.excel import UnstructuredExcelLoader
from langchain_community.document_loaders import JSONLoader

def load_all_documents(data_dir: str) -> List[Any]:
    """
    Load all supported files from the data directory and convert to LangChain document structure.
    Supported: PDF, TXT, CSV, Excel, Word, JSON
    """
    # Use project root data folder
    data_path = Path(data_dir).resolve()
    print(f"[DEBUG] Data path: {data_path}")
    documents = []

    # Exclude vector store directories if they are inside data_dir
    exclude_dirs = ["vector_store", "faiss_store", "chroma_db"]


    def is_excluded(path: Path) -> bool:
        return any(part in exclude_dirs for part in path.parts)

    # PDF files
    pdf_files = [f for f in data_path.glob('**/*.pdf') if not is_excluded(f)]
    print(f"[DEBUG] Found {len(pdf_files)} PDF files: {[str(f) for f in pdf_files]}")
    for pdf_file in pdf_files:
        print(f"[DEBUG] Loading PDF: {pdf_file}")
        try:
            loader = PyPDFLoader(str(pdf_file))
            loaded = loader.load()
            print(f"[DEBUG] Loaded {len(loaded)} PDF docs from {pdf_file}")
            documents.extend(loaded)
        except Exception as e:
            print(f"[ERROR] Failed to load PDF {pdf_file}: {e}")

    # TXT files
    txt_files = [f for f in data_path.glob('**/*.txt') if not is_excluded(f)]
    print(f"[DEBUG] Found {len(txt_files)} TXT files: {[str(f) for f in txt_files]}")
    for txt_file in txt_files:
        print(f"[DEBUG] Loading TXT: {txt_file}")
        try:
            loader = TextLoader(str(txt_file))
            loaded = loader.load()
            print(f"[DEBUG] Loaded {len(loaded)} TXT docs from {txt_file}")
            documents.extend(loaded)
        except Exception as e:
            print(f"[ERROR] Failed to load TXT {txt_file}: {e}")

    # CSV files
    csv_files = [f for f in data_path.glob('**/*.csv') if not is_excluded(f)]
    print(f"[DEBUG] Found {len(csv_files)} CSV files: {[str(f) for f in csv_files]}")
    for csv_file in csv_files:
        print(f"[DEBUG] Loading CSV: {csv_file}")
        try:
            loader = CSVLoader(str(csv_file))
            loaded = loader.load()
            print(f"[DEBUG] Loaded {len(loaded)} CSV docs from {csv_file}")
            documents.extend(loaded)
        except Exception as e:
            print(f"[ERROR] Failed to load CSV {csv_file}: {e}")

    # Excel files
    xlsx_files = [f for f in data_path.glob('**/*.xlsx') if not is_excluded(f)]
    print(f"[DEBUG] Found {len(xlsx_files)} Excel files: {[str(f) for f in xlsx_files]}")
    for xlsx_file in xlsx_files:
        print(f"[DEBUG] Loading Excel: {xlsx_file}")
        try:
            loader = UnstructuredExcelLoader(str(xlsx_file))
            loaded = loader.load()
            print(f"[DEBUG] Loaded {len(loaded)} Excel docs from {xlsx_file}")
            documents.extend(loaded)
        except Exception as e:
            print(f"[ERROR] Failed to load Excel {xlsx_file}: {e}")

    # Word files
    docx_files = [f for f in data_path.glob('**/*.docx') if not is_excluded(f)]
    print(f"[DEBUG] Found {len(docx_files)} Word files: {[str(f) for f in docx_files]}")
    for docx_file in docx_files:
        print(f"[DEBUG] Loading Word: {docx_file}")
        try:
            loader = Docx2txtLoader(str(docx_file))
            loaded = loader.load()
            print(f"[DEBUG] Loaded {len(loaded)} Word docs from {docx_file}")
            documents.extend(loaded)
        except Exception as e:
            print(f"[ERROR] Failed to load Word {docx_file}: {e}")

    # JSON files
    json_files = [f for f in data_path.glob('**/*.json') if not is_excluded(f)]
    print(f"[DEBUG] Found {len(json_files)} JSON files: {[str(f) for f in json_files]}")
    for json_file in json_files:
        print(f"[DEBUG] Loading JSON: {json_file}")
        try:
            loader = JSONLoader(str(json_file))
            loaded = loader.load()
            print(f"[DEBUG] Loaded {len(loaded)} JSON docs from {json_file}")
            documents.extend(loaded)
        except Exception as e:
            print(f"[ERROR] Failed to load JSON {json_file}: {e}")


    print(f"[DEBUG] Total loaded documents: {len(documents)}")
    return documents

# Example usage
if __name__ == "__main__":
    docs = load_all_documents("data")
    print(f"Loaded {len(docs)} documents.")
    print("Example document:", docs[0] if docs else None)