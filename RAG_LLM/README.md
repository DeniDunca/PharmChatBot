# RAG System with ASI One Mini

This is a Retrieval-Augmented Generation (RAG) system that uses the ASI One Mini model to answer questions based on PDF documents.

## Setup

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Make sure you have Ollama installed and running on your system.

3. Place your PDF documents in the same directory as the script.

## Usage

Run the script:
```bash
python rag_system.py
```

The script will:
1. Load and process the PDF documents
2. Create embeddings and store them in a vector database
3. Answer example questions about the documents

## Customization

You can modify the script to:
- Use different PDF documents by changing the `pdf_paths` list in the `main()` function
- Ask different questions by modifying the `questions` list
- Adjust the chunk size and overlap in the `load_pdf()` function
- Change the number of retrieved documents (k) in the `answer_question()` function

## Example Questions

The script includes example questions about:
- Active ingredients in Advil
- Side effects of Fluorouracil
- Storage conditions for Advil


ollama run llama3
ollama pull nomic-embed-text
