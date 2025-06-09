import json
import requests
import os
import pandas as pd
import tempfile
from dotenv import load_dotenv
from pypdf import PdfReader
from pathlib import Path
from datetime import datetime

load_dotenv()

MODEL = "asi1-mini"
URL = "https://api.asi1.ai/v1/chat/completions"
API_KEY = os.getenv('ASI_ONE_KEY')

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def save_progress(results, output_path):
    """Save current progress to JSON file."""
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

def download_pdf(url, temp_dir):
    """Download PDF from URL and save to temporary file."""
    print(f"Downloading PDF from {url}...")
    response = requests.get(url)
    if response.status_code == 200:
        # Create a temporary file with .pdf extension
        temp_file = tempfile.NamedTemporaryFile(suffix='.pdf', delete=False, dir=temp_dir)
        temp_file.write(response.content)
        temp_file.close()
        print("Download complete.")
        return temp_file.name
    else:
        raise Exception(f"Failed to download PDF from {url}: {response.status_code}")

def extract_pdf_text(pdf_path, max_chars=6000):
    """Extract text from a PDF file, optionally truncating to max_chars."""
    print("Extracting text from PDF...")
    reader = PdfReader(pdf_path)
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    print(f"Extracted {len(text)} characters of text.")
    return text[:max_chars]

def generate_questions(pdf_text):
    """Generate questions about the PDF content using ASI."""
    print("\nGenerating questions...")
    prompt = f"""Based on the following text, generate 10 relevant questions that can be answered using only this information. 
    Format the response as a JSON array of strings, each string being a question.
    Text: {pdf_text[:2000]}  # Using first 2000 chars for question generation
    """
    
    messages = [{"role": "user", "content": prompt}]
    response = call_asi_one_chatbot(messages, tokens=1000)
    
    try:
        # Try to parse the response as JSON
        questions = json.loads(response)
        if not isinstance(questions, list) or len(questions) != 10:
            raise ValueError("Response is not a list of 10 questions")
        print(f"Generated {len(questions)} questions.")
        return questions
    except:
        # If parsing fails, try to extract questions from text
        questions = [q.strip() for q in response.split('\n') if q.strip() and '?' in q]
        print(f"Extracted {len(questions)} questions from text response.")
        return questions[:10]  # Return first 10 questions found

def call_asi_one_chatbot(messages, tokens, pdf_context=None):
    """Call ASI One chatbot with optional PDF context."""
    if pdf_context:
        # Add system message to instruct the model to use only the provided context
        system_message = {
            "role": "system",
            "content": "You are a helpful assistant that answers questions based ONLY on the provided context. If the answer cannot be found in the context, respond with 'I don't know'. Do not use any external knowledge."
        }
        messages = [system_message] + messages
        # Add context to the last user message
        last_user_msg = messages[-1]
        last_user_msg["content"] = f"Context: {pdf_context}\n\nQuestion: {last_user_msg['content']}"
        messages[-1] = last_user_msg

    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {API_KEY}'
    }
    payload = json.dumps({
        "model": MODEL,
        "messages": messages,
        "temperature": 0.2,
        "top_p": 1.0,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "max_tokens": tokens,
        "stream": False
    })
    response = requests.post(URL, headers=headers, data=payload)
    if response.status_code == 200:
        return response.json().get("choices", [{}])[0].get("message", {}).get("content", "No response")
    else:
        return f"Error: {response.status_code}, {response.text}"

def chat_with_pdf_context(pdf_path, question, tokens=1000):
    """Chat with the model using PDF context."""
    # Extract all text from the PDF (or up to max_chars)
    context = extract_pdf_text(pdf_path)
    # Prepare messages
    messages = [
        {"role": "user", "content": question}
    ]
    # Call the chatbot with context
    return call_asi_one_chatbot(messages, tokens, pdf_context=context)

def process_dataset(csv_path, output_json_path):
    """Process the dataset CSV and generate Q&A pairs for each PDF."""
    # Read the CSV file
    print(f"Reading dataset from {csv_path}")
    df = pd.read_csv(csv_path)
    total_pdfs = len(df)
    
    # Load existing results if any
    if os.path.exists(output_json_path):
        print(f"Loading existing results from {output_json_path}")
        with open(output_json_path, 'r') as f:
            results = json.load(f)
        processed_ids = {r['id'] for r in results if 'error' not in r}
    else:
        results = []
        processed_ids = set()
    
    # Create a temporary directory for PDFs
    with tempfile.TemporaryDirectory() as temp_dir:
        # Process each PDF
        for index, row in df.iterrows():
            pdf_id = row['id']
            
            # Skip if already processed successfully
            if pdf_id in processed_ids:
                print(f"\nSkipping PDF {pdf_id} (already processed)")
                continue
            
            print(f"\n{'='*50}")
            print(f"Processing PDF {pdf_id} ({index + 1}/{total_pdfs})")
            print(f"Link: {row['link']}")
            print(f"{'='*50}\n")
            
            try:
                # Download PDF
                pdf_path = download_pdf(row['link'], temp_dir)
                
                # Extract text
                pdf_text = extract_pdf_text(pdf_path)
                
                # Generate questions
                questions = generate_questions(pdf_text)
                
                # Get answers for each question
                qa_pairs = []
                for q_idx, question in enumerate(questions, 1):
                    print(f"\nQuestion {q_idx}/10: {question}")
                    answer = chat_with_pdf_context(pdf_path, question)
                    print(f"Answer: {answer[:100]}...")  # Show first 100 chars of answer
                    
                    qa_pairs.append({
                        "question": question,
                        "answer": answer
                    })
                    
                    # Save progress after each Q&A pair
                    current_result = {
                        "id": pdf_id,
                        "link": row['link'],
                        "qa_pairs": qa_pairs,
                        "last_updated": datetime.now().isoformat()
                    }
                    
                    # Update or append to results
                    result_index = next((i for i, r in enumerate(results) if r['id'] == pdf_id), -1)
                    if result_index >= 0:
                        results[result_index] = current_result
                    else:
                        results.append(current_result)
                    
                    save_progress(results, output_json_path)
                    print(f"Progress saved after question {q_idx}")
                
                # Clean up the temporary PDF file
                os.unlink(pdf_path)
                print(f"\nCompleted processing PDF {pdf_id}")
                
            except Exception as e:
                print(f"\nError processing PDF {pdf_id}: {str(e)}")
                error_result = {
                    "id": pdf_id,
                    "link": row['link'],
                    "error": str(e),
                    "last_updated": datetime.now().isoformat()
                }
                
                # Update or append error result
                result_index = next((i for i, r in enumerate(results) if r['id'] == pdf_id), -1)
                if result_index >= 0:
                    results[result_index] = error_result
                else:
                    results.append(error_result)
                
                save_progress(results, output_json_path)
        
        return results

if __name__ == "__main__":
    # Use absolute paths based on script location
    csv_path = os.path.join(SCRIPT_DIR, "dataset.csv")
    output_path = os.path.join(SCRIPT_DIR, "qa_results.json")
    
    print(f"Starting processing at {datetime.now().isoformat()}")
    print(f"CSV file: {csv_path}")
    print(f"Output file: {output_path}")
    
    # Process the dataset and save results
    results = process_dataset(csv_path, output_path)
    
    print(f"\nProcessing completed at {datetime.now().isoformat()}")
    print(f"Processed {len(results)} PDFs. Results saved to {output_path}")