from sentence_transformers import SentenceTransformer, util
import numpy as np
import json
from RAG import *

model = SentenceTransformer("all-MiniLM-L6-v2")

def context_recall(gt_docs, retrieved_docs, threshold=0.7):
    gt_emb = model.encode(gt_docs, convert_to_tensor=True)
    retrieved_emb = model.encode(retrieved_docs, convert_to_tensor=True)
    sim = util.cos_sim(gt_emb, retrieved_emb)
    return (sim.max(dim=1).values > threshold).float().mean().item()

def context_precision(gt_docs, retrieved_docs, threshold=0.7):
    gt_emb = model.encode(gt_docs, convert_to_tensor=True)
    retrieved_emb = model.encode(retrieved_docs, convert_to_tensor=True)
    sim = util.cos_sim(retrieved_emb, gt_emb)
    return (sim.max(dim=1).values > threshold).float().mean().item()

def f1(p, r): return 2 * p * r / (p + r) if (p + r) else 0.0

def evaluate(retrieved_contexts, ground_truth_contexts):
    if len(retrieved_contexts) != len(ground_truth_contexts):
        raise ValueError("Retrieved contexts and ground truth contexts must have the same length.")
    
    results = []
    for gt, ret in zip(ground_truth_contexts, retrieved_contexts):
        p = context_precision(gt, ret)
        r = context_recall(gt, ret)
        results.append((p, r, f1(p, r)))
    results = np.array(results)
    results = np.mean(results, axis=0)
    return list(results)

def test_RAG(collection, embed_model, log_ground_truth, data):
    ground_truth_contexts=[]
    retrieved_contexts=[]
    print('test with only problem logs')
    for ground_truth in log_ground_truth:
        ground_truth_contexts.append(ground_truth['title'])
        retrieved_contexts.append(RAG_search(ground_truth['log'], collection, embed_model)[0]['title'])
    print(f"Number of ground truth contexts: {len(ground_truth_contexts)}")
        
    avg_p, avg_r, avg_f1 = evaluate(retrieved_contexts, ground_truth_contexts)
    print(f"Precision: {avg_p:.3f}")
    print(f"Recall:    {avg_r:.3f}")
    print(f"F1 Score:  {avg_f1:.3f}")

    print('now, test with full log')
    ground_truth_contexts=[]
    retrieved_contexts=[]
    for text in data:
        #text = log_pre_processing(text) # Already pre-processed in data
        for ground_truth in log_ground_truth:
            if  ground_truth['log'] in text:
                ground_truth_contexts.append(ground_truth['title'])
                result = RAG_search('\n'.join(text), collection, embed_model)
                retrieved_contexts.append(result[0]['title'])
                break
    print(f"Number of ground truth contexts: {len(ground_truth_contexts)}")
        
    avg_p, avg_r, avg_f1 = evaluate(retrieved_contexts, ground_truth_contexts)
    print(f"Precision: {avg_p:.3f}")
    print(f"Recall:    {avg_r:.3f}")
    print(f"F1 Score:  {avg_f1:.3f}")

if __name__ == "__main__":
    data_dir = 'evaluation_data'
    with open(f'{data_dir}/processed_log.json', 'r') as f:
        data = json.load(f)
    print(len(data))

    # RAG import
    db_list=['kubernetes_docs.db', 'ansible_docs.db', 'stackoverflow_docs.db']
    collection, embed_model = RAG_init(db_list)

    with open(f"{data_dir}/bad_log_linked.json", 'r') as f:
        log_ground_truth = json.load(f)
    print('all-MiniLM-L6-v2')
    test_RAG(collection, embed_model, log_ground_truth, data)

    print("now, let's test infly model")
    collection, embed_model = RAG_init(db_list, embed_model='infly')
    test_RAG(collection, embed_model, log_ground_truth, data)
