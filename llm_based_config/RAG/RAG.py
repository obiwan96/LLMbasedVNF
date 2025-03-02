import sqlite3
from sentence_transformers import SentenceTransformer
import chromadb
import requests
import uuid

def RAG_init(collection, db_name):
    # 모델 및 벡터DB 초기화
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')

    # DB 연결
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    # 데이터 조회
    cursor.execute("SELECT id, title, content FROM documents")
    rows = cursor.fetchall()
    print(f'DB has total {len(rows)} num of doc.s')
    documents = []
    for row in rows:
        documents.append({'id': row[0], 'title': row[1], 'text': row[2]})

    conn.close()

    # 임베딩 및 저장
    for doc in documents:
        embedding = embed_model.encode(doc["title"]+doc["text"]).tolist()
        collection.add(documents=[doc["text"]], embeddings=[embedding], metadatas=[{'title':doc['title']}], ids=[str(uuid.uuid4())])


    # 사용자 쿼리
    query = "How to create a new instance with python?"

    # 쿼리 임베딩
    query_embedding = embed_model.encode(query).tolist()

    # 벡터 검색 (유사한 문서 2개 반환)
    results = collection.query(query_embeddings=[query_embedding], n_results=4)

    # 검색된 문서 확인
    retrieved_texts_titles = [item['title'] for item in results['metadatas'][0]]
    print("검색된 문서:", retrieved_texts_titles)
    retrieved_texts = results['documents']

    # Ollama API 엔드포인트
    OLLAMA_API_URL = "http://localhost:11434/api/generate"  # 로컬에서 실행 중인 Ollama

    # LLM에 전달할 프롬프트 구성
    prompt = f"다음 정보를 참고하여 질문에 답하세요:\n\n{retrieved_texts}\n\n질문: {query}"

    # Ollama API 호출
    response = requests.post(OLLAMA_API_URL, json={
        "model": "llama2",  # 사용 중인 모델 이름
        "prompt": prompt,
        "stream": False
    })

    # 결과 출력
    answer = response.json().get("response")
    print("LLM의 답변:", answer)

if __name__ == '__main__':
    chroma_client = chromadb.Client()
    collection = chroma_client.create_collection("documents")
    db_name = 'openstack_docs_v2.db'
    RAG_init(collection, db_name)