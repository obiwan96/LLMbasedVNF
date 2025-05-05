import sqlite3
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from sentence_transformers import SentenceTransformer
import chromadb
import uuid

logging_file_rag = 'log_rag.txt'
def RAG_init(db_names): 
    # db_names msut me a list of db_names. ex) ['kubernetes_docs.db']
    collection_name="documents"
    chroma_client = chromadb.Client()
    if collection_name in chroma_client.list_collections():
        chroma_client.delete_collection(name=collection_name)    
    collection = chroma_client.create_collection(collection_name)
    #embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    # Let's use INF-Retriever-v1 model, which shows higher performance in coding domain
    embed_model = SentenceTransformer("infly/inf-retriever-v1", trust_remote_code=True)

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
                if row[2].strip() != "":
                    documents.append({'id': row[0], 'title': row[1], 'text': row[2], 'question': row[3]})
        else:
            # Question colum dosen't exist.
            cursor.execute("SELECT id, title, content FROM documents")
            rows = cursor.fetchall()
            for row in rows:
                if row[2].strip() != '':
                    documents.append({'id': row[0], 'title': row[1], 'text': row[2], 'question':row[2]}) 
                    # To make embedding, I will use title and text for these cases.
                    # If question exist, use tite and question for embedding
        
        conn.close()
    print('RAG Init.')
    print(f'DB has total {len(documents)} num of doc.s')
    
    for doc in documents:
        if doc["question"]:
            embedding = embed_model.encode(doc["title"]+doc["question"]).tolist()
            collection.add(documents=[doc["question"]+doc["text"]], embeddings=[embedding], metadatas=[{'title':doc['title']}], ids=[str(uuid.uuid4())])
        else:
            embedding = embed_model.encode(doc["title"]).tolist()
            collection.add(documents=[doc["text"]], embeddings=[embedding], metadatas=[{'title':doc['title']}], ids=[str(uuid.uuid4())])
    return collection, embed_model

def RAG_search(query, collection, embed_model, logging_=False, n_results=1, vnf_name=None):
    # vector search only now.
    # Todo: Graph based search
    if 'exit code' in str(query):
        return ''
    query_embedding = embed_model.encode(str(query)).tolist()

    results = collection.query(query_embeddings=[query_embedding], n_results=n_results)

    retrieved_texts_titles = [item['title'] for item in results['metadatas'][0]]
    #print("Retrieved:", retrieved_texts_titles)
    retrieved_texts = results['documents'][0][0]
    if logging_:
        with open(logging_file_rag, 'a') as f:
            f.write('------------------------------\n')
            if vnf_name:
                f.write('VNF name: '+vnf_name+'\n')
            f.write('RAG input:\n')
            f.write(str(query)+'\n')
            f.write('#####\n')
            f.write('RAG results:\n')
            f.write('\n'.join(retrieved_texts_titles)+'\n')
            f.write(retrieved_texts+'\n')
    if retrieved_texts.strip() == '':
        print('RAG something wrong'+ retrieved_texts_titles[0])
    return '\nAnd here is a related document. Please refer to it.\n' + retrieved_texts

if __name__ == '__main__':
    db_name = 'openstack_docs.db'
    RAG_init(db_name)