import sqlite3
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from sentence_transformers import SentenceTransformer
import chromadb
import uuid
import re

logging_file_rag = 'log_rag.txt'

def remove_excess_escapes(log: str) -> str:
    # backslash remove
    log = re.sub(r'\\{2,}', r'\\', log)
    try:
        log = bytes(log, 'utf-8').decode('unicode_escape')
    except Exception:
        pass
    return log

def clean_garbage_chars(text: str) -> str:
    # some ascii code crack. remove
    # 1. Non-breaking space, unexpected Latin characters
    text = text.replace('\xa0', ' ')  # NBSP 제거
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # ASCII 이외 문자 제거 (or replace with ' ')
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def log_pre_processing(log):
    log = remove_excess_escapes(log)
    lines = log.split('\n')
    pre_processed_log = []
    for line in lines:
        if any(skip in line for skip in ['[WARNING]', 'PLAY ', 'TASK ', 'RECAP', 'ok=', 'changed=']):
            continue
        pre_processed_log.append(clean_garbage_chars(line.strip()))
    return '\n'.join(pre_processed_log)

def RAG_init(db_names, embed_model='all-MiniLM-L6-v2', new = False): 
    # db_names msut me a list of db_names. ex) ['kubernetes_docs.db']
    collection_name="documents-"+embed_model
    chroma_client = chromadb.Client()
    if embed_model=='infly':
        # Let's use INF-Retriever-v1 model, which shows higher performance in coding domain
        # But after change, initiation becomes much slower.
        embed_model = SentenceTransformer("infly/inf-retriever-v1", trust_remote_code=True)
        embed_model.max_seq_length = 512 # embedding model get title and question, so 512 is enough.
    elif embed_model =='fine-tuned':
        embed_model = SentenceTransformer('fine_tuned_model/all-MiniLM-L6-v2-finetuned')
    else:
        embed_model = SentenceTransformer(embed_model)
    if collection_name in chroma_client.list_collections():
        if new:
            chroma_client.delete_collection(name=collection_name)
        else:
            collection = chroma_client.get_collection(name=collection_name)
            return collection, embed_model
    collection = chroma_client.create_collection(collection_name)
    
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
                    documents.append({'id': row[0], 'title': row[1], 'text': row[2], 'question': row[3], 'db_name': db_name})
        else:
            # Question colum dosen't exist.
            cursor.execute("SELECT id, title, content FROM documents")
            rows = cursor.fetchall()
            for row in rows:
                if row[2].strip() != '':
                    documents.append({'id': row[0], 'title': row[1], 'text': row[2], 'question':row[2], 'db_name': db_name}) 
                    # To make embedding, I will use title and text for these cases.
                    # If question exist, use tite and question for embedding
        
        conn.close()
    print('RAG Init.')
    print(f'DB has total {len(documents)} num of doc.s')
    
    for doc in documents:
        if doc["question"]:
            embedding = embed_model.encode(doc["title"]+doc["question"]).tolist()
            collection.add(documents=[doc["question"]+doc["text"]], embeddings=[embedding], metadatas=[{'title':doc['title'], 'db_name': doc['db_name']}], ids=[str(uuid.uuid4())])
        else:
            embedding = embed_model.encode(doc["title"]).tolist()
            collection.add(documents=[doc["text"]], embeddings=[embedding], metadatas=[{'title':doc['title'], 'db_name': doc['db_name']}], ids=[str(uuid.uuid4())])
    return collection, embed_model

def RAG_search(query, collection, embed_model, logging_=False, n_results=1, vnf_name=None):
    # vector search only now.
    # Todo: Graph based search
    '''if 'exit code' in str(query):
        return '' '''# It returned nothing if container exit code is delivered. not need from now on. it's related to errorcode with 33.    

    query = log_pre_processing(query)
    query_embedding = embed_model.encode(str(query)).tolist()

    results = collection.query(query_embeddings=[query_embedding], n_results=n_results)
    
    retrieved_texts = []
    for i in range(n_results):
        if results['documents'][0][i] in [None, '']:
            break
        retrieved_texts.append({
            'title': results['metadatas'][0][i]['title'],
            'text': results['documents'][0][i],
            'db_name': results['metadatas'][0][i]['db_name'],
            'distance': results['distances'][0][i]
        })

    #retrieved_texts_titles = [item['title'] for item in results['metadatas'][0]]
    #print("Retrieved:", retrieved_texts_titles)
    #retrieved_texts = results['documents'][0]
    if logging_:
        with open(logging_file_rag, 'a') as f:
            f.write('*******-------------------******\n')
            if vnf_name:
                f.write('VNF name: '+vnf_name+'\n')
            f.write('RAG input:\n')
            f.write('RAG results:\n')
            f.write(str(query)+'\n')
            for i in range(len(retrieved_texts)):
                f.write('------------------------------\n')              
                f.write('Title: '+retrieved_texts[i]['title']+'\n')
                f.write('Distance: '+str(retrieved_texts[i]['distance'])+'\n')
                f.write('DB name: '+retrieved_texts[i]['db_name']+'\n')
                f.write('Text: '+retrieved_texts[i]['text']+'\n')
    if len(retrieved_texts) == 0:
        print('RAG something wrong')
    #return '\nAnd here is a related document. Please refer to it.\n' + retrieved_texts
    
    ##### TODO: change the return format, so need to change the main.py ##########
    return retrieved_texts

def color_text(text, color='red'):
    colors = {
        'red': '\033[91m',
        'green': '\033[92m',
        'yellow': '\033[93m',
        'blue': '\033[94m',
        'magenta': '\033[95m',
        'cyan': '\033[96m',
        'white': '\033[97m',
        'reset': '\033[0m'
    }
    return f"{colors.get(color, colors['reset'])}{text}{colors['reset']}"

if __name__ == '__main__':
    db_list=['kubernetes_docs.db', 'ansible_docs.db', 'stackoverflow_docs.db']
    from docs_crawler import delete_overlap
    for db_name in db_list:
        delete_overlap(db_name)
    collection, embed_model = RAG_init(db_list)
    print(color_text('RAG init end', 'red'))
    while True:
        query = input('Enter your query: ')
        if query.lower() == 'exit':
            break
        result = RAG_search(query, collection, embed_model)
        for res in result:
            print(color_text(f"Title: {res['title']}, Distance: {res['distance']}, DB Name: {res['db_name']}", 'green') )
            print(f"Text: {res['text']}\n")