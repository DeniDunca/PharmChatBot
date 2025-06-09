# 1. Ranking (Re-Rank, RankGPT, RAG-Fusion)
def ranking(question, documents, rank_fn):
    """
    Rank or filter documents based on their relevance to the question using a ranking function.
    """
    ranked = sorted(documents, key=lambda doc: rank_fn(question, doc), reverse=True)
    return ranked

# 2. Refinement (CRAG)
def refinement(documents, refine_fn):
    """
    Refine, filter, or compress documents based on relevance or quality using a refinement function (e.g., CRAG).
    """
    return [refine_fn(doc) for doc in documents]

# 3. Active Retrieval
def active_retrieval(question, initial_docs, retrieve_fn, threshold=0.5):
    """
    Re-retrieve and/or retrieve from new data sources if initial documents are not relevant enough.
    """
    relevant_docs = [doc for doc in initial_docs if retrieve_fn(question, doc) > threshold]
    if not relevant_docs:
        # Simulate retrieving from a new data source
        return [f"New doc for: {question}"]
    return relevant_docs

# --- Test Example ---
if __name__ == "__main__":
    question = "What are the side effects of Atripla?"
    docs = [
        "Atripla may cause nausea and headache.",
        "Consult your doctor before use.",
        "Atripla is used to treat HIV.",
    ]

    def dummy_rank_fn(q, doc):
        # Simple ranking: count shared words
        return sum(1 for word in q.lower().split() if word in doc.lower())

    def dummy_refine_fn(doc):
        # Just return the first sentence
        return doc.split('.')[0] + '.'

    def dummy_retrieve_fn(q, doc):
        # Return 1.0 if any keyword matches, else 0.0
        keywords = ["side effects", "nausea", "headache"]
        return 1.0 if any(k in doc.lower() for k in keywords) else 0.0

    print("Ranking:", ranking(question, docs, dummy_rank_fn))
    print("\nRefinement:", refinement(docs, dummy_refine_fn))
    print("\nActive Retrieval:", active_retrieval(question, docs, dummy_retrieve_fn))
