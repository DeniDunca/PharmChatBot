import numpy as np

class Routing:
    def __init__(self, logical_rules=None, semantic_prompts=None, embed_fn=None):
        """
        logical_rules: list of (keyword, data_source) tuples
        semantic_prompts: dict of {prompt_name: prompt_text}
        embed_fn: function to embed text (for semantic routing)
        """
        self.logical_rules = logical_rules or []
        self.semantic_prompts = semantic_prompts or {}
        self.embed_fn = embed_fn or self.dummy_embed
        self.prompt_embeddings = {k: self.embed_fn(v) for k, v in self.semantic_prompts.items()}

    def logical_route(self, question):
        """Route based on keyword rules."""
        for keyword, data_source in self.logical_rules:
            if keyword.lower() in question.lower():
                return data_source
        return "default"

    def semantic_route(self, question):
        """Route based on embedding similarity to prompts."""
        q_emb = self.embed_fn(question)
        best_prompt = None
        best_score = -float('inf')
        for name, emb in self.prompt_embeddings.items():
            score = np.dot(q_emb, emb) / (np.linalg.norm(q_emb) * np.linalg.norm(emb))
            if score > best_score:
                best_score = score
                best_prompt = name
        return best_prompt

    @staticmethod
    def dummy_embed(text):
        # Simple embedding: vector of character ords (padded/truncated)
        arr = np.zeros(16)
        for i, c in enumerate(text[:16]):
            arr[i] = ord(c)
        return arr

# Test example
if __name__ == "__main__":
    logical_rules = [
        ("how many", "sql_db"),
        ("interact", "graph_db"),
        ("find", "vector_db"),
    ]
    semantic_prompts = {
        "side_effects": "What are the side effects of this medication?",
        "usage": "What is this medication used for?",
        "warnings": "Are there any warnings for this medication?",
    }
    router = Routing(logical_rules=logical_rules, semantic_prompts=semantic_prompts)

    questions = [
        "How many patients were prescribed Atripla?",
        "Show me all drugs that interact with Atripla.",
        "Find clinical trial reports about Atripla.",
        "What should I avoid when taking Atripla?",
        "What is Atripla used for?",
    ]

    print("Logical Routing Results:")
    for q in questions:
        print(f"Q: {q}\n  -> {router.logical_route(q)}")

    print("\nSemantic Routing Results:")
    for q in questions:
        print(f"Q: {q}\n  -> {router.semantic_route(q)}")
