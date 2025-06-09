from asi_chat import call_asi_one_chatbot

def text_to_sql(question, table_schema):
    prompt = (
        f"You are an expert in SQL. Given the following table schema:\n{table_schema}\n"
        f"Convert this question to an SQL query: {question}\n"
        "Return only the SQL query."
    )
    messages = [{"role": "user", "content": prompt}]
    sql_query = call_asi_one_chatbot(messages, tokens=200)
    return sql_query.strip()

def text_to_cypher(question, graph_schema):
    prompt = (
        f"You are an expert in Cypher (Neo4j). Given the following graph schema:\n{graph_schema}\n"
        f"Convert this question to a Cypher query: {question}\n"
        "Return only the Cypher query."
    )
    messages = [{"role": "user", "content": prompt}]
    cypher_query = call_asi_one_chatbot(messages, tokens=200)
    return cypher_query.strip()

def self_query_retriever(question, metadata_schema):
    prompt = (
        f"You are an expert in semantic search. Given the following metadata schema:\n{metadata_schema}\n"
        f"Extract the relevant filters from this question for a vector search: {question}\n"
        "Return the filters as a JSON object."
    )
    messages = [{"role": "user", "content": prompt}]
    filters = call_asi_one_chatbot(messages, tokens=200)
    return filters.strip()

if __name__ == "__main__":
    # Example schemas
    table_schema = "Table: prescriptions (id, patient_id, drug, year, dosage)"
    graph_schema = "Nodes: Drug, Patient; Relationships: PRESCRIBED, INTERACTS_WITH"
    metadata_schema = "Fields: drug, year, document_type"

    question = "How many patients were prescribed Atripla last year?"

    print("SQL Query:", text_to_sql(question, table_schema))
    print("Cypher Query:", text_to_cypher(question, graph_schema))
    print("Vector DB Filters:", self_query_retriever(question, metadata_schema))
