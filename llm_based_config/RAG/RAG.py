import sqlite3
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from sentence_transformers import SentenceTransformer
import chromadb
import uuid

def RAG_init(db_name):
    chroma_client = chromadb.Client()
    collection = chroma_client.create_collection("documents")
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')

    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    cursor.execute("SELECT id, title, content FROM documents")
    rows = cursor.fetchall()
    print(f'DB has total {len(rows)} num of doc.s')
    documents = []
    for row in rows:
        if not row[2] == '':
            documents.append({'id': row[0], 'title': row[1], 'text': row[2]})

    conn.close()

    for doc in documents:
        embedding = embed_model.encode(doc["title"]+doc["text"]).tolist()
        collection.add(documents=[doc["text"]], embeddings=[embedding], metadatas=[{'title':doc['title']}], ids=[str(uuid.uuid4())])
    return collection, embed_model

def RAG_search(query, collection, embed_model, n_results=1):
    query_embedding = embed_model.encode(query).tolist()

    results = collection.query(query_embeddings=[query_embedding], n_results=n_results)

    retrieved_texts_titles = [item['title'] for item in results['metadatas'][0]]
    #print("Retrieved:", retrieved_texts_titles)
    retrieved_texts = results['documents'][0][0]
    return retrieved_texts

if __name__ == '__main__':
    db_name = 'openstack_docs.db'
    RAG_init(db_name)