import requests
import os
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import DocArrayInMemorySearch
import requests
from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser
import pandas as pd
import requests
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from huggingface_hub import login

load_dotenv()

MODEL = "asi1-mini"
URL = "https://api.asi1.ai/v1/chat/completions"
API_KEY = os.getenv('ASI_ONE_KEY')



os.makedirs("pdfs", exist_ok=True)

df = pd.read_csv(os.path.join(os.path.dirname(__file__), "dataset.csv"))
links = df['link'].tolist()

pdf_files = []

for i, link in enumerate(links):
    filename = f"pdfs/doc_{i}.pdf"
    try:
        r = requests.get(link)
        if r.ok:
            with open(filename, "wb") as f:
                f.write(r.content)
            pdf_files.append(filename)
            print(f"Downloaded {filename}")
        else:
            print(f"Failed to download {link}")
    except Exception as e:
        print(f"Error downloading {link}: {e}")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

all_documents = []

for filepath in pdf_files:
    try:
        loader = PyPDFLoader(filepath)
        docs = loader.load_and_split(text_splitter)
        all_documents.extend(docs)
        print(f"Parsed {filepath} with {len(docs)} chunks.")
    except Exception as e:
        print(f"Error parsing {filepath}: {e}")



login(token=os.getenv("TOKEN"))

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L3-v2")  

vectorstore = DocArrayInMemorySearch.from_documents(all_documents, embedding=embedding_model)
print(vectorstore)

template = """
Answer the question based on the context below. If you can't
answer the question, reply "I don't know".

Context: {context}

Question: {question}
"""

prompt = PromptTemplate.from_template(template)
print(prompt.format(context="Here is some context", question="Here is a question"))


def call_asi_one(prompt):
    if hasattr(prompt, "to_string"):
        prompt = prompt.to_string()  # Convert PromptValue to str

    url = "https://api.asi1.ai/v1/chat/completions"
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {API_KEY}'
    }
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "max_tokens": 1000
    }
    response = requests.post(url, headers=headers, json=payload)
    # Log the response status code and content
    print(f"Response Status Code: {response.status_code}")
    print(f"Response Content: {response.text}")
    # Check if the response is successful
    if response.status_code == 200:
        return response.json().get("choices", [{}])[0].get("message", {}).get("content", "No response")
    else:
        return f"Error: {response.status_code}, {response.text}"
# Create the RAG chain
chain = ({
    "context": itemgetter("question") | vectorstore.as_retriever(),
    "question": itemgetter("question"),
}
| prompt
| call_asi_one
| StrOutputParser()
)