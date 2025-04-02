import sqlite3
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from sentence_transformers import SentenceTransformer
import chromadb
import uuid

def RAG_init(db_names): 
    # db_names msut me a list of db_names. ex) ['kubernetes_docs.db']
    collection_name="documents"
    chroma_client = chromadb.Client()
    if collection_name in chroma_client.list_collections():
        chroma_client.delete_collection(name=collection_name)    
    collection = chroma_client.create_collection(collection_name)
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')

    documents = []
    for db_name in db_names:
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()
        cursor.execute(f"PRAGMA table_info(documents);")
        columns = cursor.fetchall()
        question_exists = any(col[1] == 'question' for col in columns)
        if question_exists:
            cursor.execute("SELECT id, title, content, question FROM documents")
            rows = cursor.fetchall()
            for row in rows:
                documents.append({'id': row[0], 'title': row[1], 'text': row[2], 'question': row[3]})
        else:
            # Question colum dosen't exist.
            cursor.execute("SELECT id, title, content FROM documents")
            rows = cursor.fetchall()
            for row in rows:
                if not row[2] == '':
                    documents.append({'id': row[0], 'title': row[1], 'text': row[2], 'question':row[2]}) 
                    # To make embedding, I will use title and text for these cases.
                    # If question exist, use tite and question for embedding
        
        conn.close()
    print('RAG Init.')
    print(f'DB has total {len(documents)} num of doc.s')
    
    for doc in documents:
        if doc["question"]:
            embedding = embed_model.encode(doc["title"]+doc["question"]).tolist()
        else:
            embedding = embed_model.encode(doc["title"]).tolist()
        collection.add(documents=[doc["text"]], embeddings=[embedding], metadatas=[{'title':doc['title']}], ids=[str(uuid.uuid4())])
    return collection, embed_model

def RAG_search(query, collection, embed_model, n_results=1):
    # vector search only now.
    # Todo: Graph based search
    query_embedding = embed_model.encode(str(query)).tolist()

    results = collection.query(query_embeddings=[query_embedding], n_results=n_results)

    retrieved_texts_titles = [item['title'] for item in results['metadatas'][0]]
    #print("Retrieved:", retrieved_texts_titles)
    retrieved_texts = results['documents'][0][0]
    return retrieved_texts

if __name__ == '__main__':
    db_name = 'openstack_docs.db'
    RAG_init(db_name)