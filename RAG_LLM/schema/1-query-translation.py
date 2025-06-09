import os
import tempfile
import pandas as pd
from asi_chat import download_pdf, extract_pdf_text, call_asi_one_chatbot

# Query Translation Methods

def multi_query(question):
    """Generate multiple rephrasings of the question."""
    prompt = f"Rephrase the following question in 3 different ways, keeping the meaning the same. Return as a JSON list. Question: {question}"
    messages = [{"role": "user", "content": prompt}]
    response = call_asi_one_chatbot(messages, tokens=300)
    try:
        return pd.read_json(response, typ='series').tolist()
    except:
        return [q.strip() for q in response.split('\n') if q.strip()]

def rag_fusion(question):
    """Generate 3 variations of the question for RAG-Fusion."""
    prompt = f"Generate 3 semantically different versions of the following question for use in RAG-Fusion. Return as a JSON list. Question: {question}"
    messages = [{"role": "user", "content": prompt}]
    response = call_asi_one_chatbot(messages, tokens=300)
    try:
        return pd.read_json(response, typ='series').tolist()
    except:
        return [q.strip() for q in response.split('\n') if q.strip()]

def decomposition(question):
    """Decompose a complex question into simpler sub-questions."""
    prompt = f"Break down the following question into 2-3 simpler sub-questions. Return as a JSON list. Question: {question}"
    messages = [{"role": "user", "content": prompt}]
    response = call_asi_one_chatbot(messages, tokens=300)
    try:
        return pd.read_json(response, typ='series').tolist()
    except:
        return [q.strip() for q in response.split('\n') if q.strip()]

def step_back(question):
    """Reformulate the question to a more general or abstract form."""
    prompt = f"Reformulate the following question to a more general or abstract version. Return as a string. Question: {question}"
    messages = [{"role": "user", "content": prompt}]
    response = call_asi_one_chatbot(messages, tokens=100)
    return response.strip()

def hyde(question):
    """Convert the question into a hypothetical answer or document for HyDE."""
    prompt = f"Write a hypothetical answer or document that would answer the following question. Question: {question}"
    messages = [{"role": "user", "content": prompt}]
    response = call_asi_one_chatbot(messages, tokens=300)
    return response.strip()

# Test with the first element from dataset.csv
if __name__ == "__main__":
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), "dataset.csv"))
    first_link = df.iloc[0]['link']
    with tempfile.TemporaryDirectory() as temp_dir:
        pdf_path = download_pdf(first_link, temp_dir)
        pdf_text = extract_pdf_text(pdf_path)
        # Use the first 200 chars as a sample question context, but make it patient-specific
        sample_question = (
            "Imi poti psune ce este atripla si la ce ajuta? Cand nu ar trebui sa o iau? "
            f"Aici e niste context: {pdf_text[:200]}"
        )
        print("Sample question:", sample_question)
        print("\nMulti-query:", multi_query(sample_question))
        print("\nRAG-Fusion:", rag_fusion(sample_question))
        print("\nDecomposition:", decomposition(sample_question))
        print("\nStep-back:", step_back(sample_question))
        print("\nHyDE:", hyde(sample_question))
