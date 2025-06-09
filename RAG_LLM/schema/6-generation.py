# 1. Active Retrieval (Self-RAG, RRR)
def self_rag(question, retrieved_docs, generate_fn):
    """
    Use generation quality to inform question re-writing and/or re-retrieval of documents.
    This method rewrites the question or triggers a new retrieval if the answer quality is low.
    """
    # Generate an answer from the retrieved documents
    answer = generate_fn(question, retrieved_docs)
    # Placeholder: if answer is too short, rewrite question and try again
    if len(answer) < 30:
        new_question = f"Rewrite: {question} (be more specific)"
        answer = generate_fn(new_question, retrieved_docs)
    return answer

# 2. RRR (Re-Retrieve, Re-Rank, Re-Generate)
def rrr(question, retrieve_fn, rank_fn, generate_fn):
    """
    Re-retrieve, re-rank, and re-generate the answer if the initial answer is not satisfactory.
    """
    docs = retrieve_fn(question)
    ranked_docs = sorted(docs, key=lambda doc: rank_fn(question, doc), reverse=True)
    answer = generate_fn(question, ranked_docs)
    # Placeholder: if answer is too short, re-retrieve and try again
    if len(answer) < 30:
        docs = retrieve_fn(f"More details: {question}")
        ranked_docs = sorted(docs, key=lambda doc: rank_fn(question, doc), reverse=True)
        answer = generate_fn(question, ranked_docs)
    return answer

# --- Test Example ---
if __name__ == "__main__":
    question = "What are the side effects of Atripla?"
    docs = [
        "Atripla may cause nausea and headache.",
        "Consult your doctor before use.",
        "Atripla is used to treat HIV.",
    ]

    def dummy_generate_fn(q, docs):
        # Just join the most relevant doc
        return docs[0] if docs else "No answer found."

    def dummy_retrieve_fn(q):
        # Return all docs for simplicity
        return docs

    def dummy_rank_fn(q, doc):
        # Simple ranking: count shared words
        return sum(1 for word in q.lower().split() if word in doc.lower())

    print("Self-RAG:", self_rag(question, docs, dummy_generate_fn))
    print("\nRRR:", rrr(question, dummy_retrieve_fn, dummy_rank_fn, dummy_generate_fn))
