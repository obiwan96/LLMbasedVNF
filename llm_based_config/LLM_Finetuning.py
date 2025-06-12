from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, AutoModelForSequenceClassification
from peft import get_peft_model, LoraConfig, TaskType, PeftModel, PeftConfig
from datasets import Dataset
from trl import SFTTrainer
import json
from RAG import RAG
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
import torch
import random 
os.environ["TRANSFORMERS_CACHE"]        = "/storage1/hf_cache/transformers"

def unwrap(x, want="tokenizer"):
    if isinstance(x, tuple):
        return x[1] if want == "tokenizer" else x[0]
    if isinstance(x, dict):
        return x[want]
    return x

def prepare_tokenizer(name: str):
    tok = unwrap(AutoTokenizer.from_pretrained(name, use_fast=True,
        trust_remote_code=True))
    if tok.eos_token is None:                       # EOS 없으면 추가
        tok.add_special_tokens({"eos_token": "</s>"})
    if tok.pad_token is None:                       # PAD 없으면 EOS와 동일
        tok.pad_token      = tok.eos_token
        tok.pad_token_id   = tok.eos_token_id
    tok.padding_side = "right"
    return tok

def load_clm(name: str, **kw):
    return unwrap(AutoModelForCausalLM.from_pretrained(name, **kw), "model")

def load_scm(name: str, **kw):
    return unwrap(AutoModelForSequenceClassification.from_pretrained(name, **kw),
                  "model")

def tokenize(example):
    full_text = example["prompt"] + example["completion"]
    tokenized = tokenizer(
        full_text,
        truncation=True,
        max_length=512,
        padding="max_length"
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

model_name = "google/gemma-3-4b-it"  
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token  # 필요시 설정
data_file = "RAG/evaluation_data/bad_log_linked.json"
model_save_path = 'finetuned_llms/'

# RAG initialization
db_list=['RAG/kubernetes_docs.db', 'RAG/ansible_docs.db', 'RAG/stackoverflow_docs.db']
collection, embed_model = RAG.RAG_init(db_list, new = True)
all_docs = collection.get(include=["embeddings", "documents", "metadatas"])
doc_vectors = np.array(all_docs['embeddings'])
doc_texts = all_docs['documents']
doc_titles = [doc['title'] for doc in all_docs['metadatas']]

with open(data_file, 'r') as f:
    data = json.load(f)
inputs = []
outputs = []
mismatched_titles = []
for linked_log in data:
    error_log = linked_log['log']
    result = linked_log['title']
    #print(f"Processing error log: {error_log}")
    #print(f"Expected title: {result}")
    query_vector = embed_model.encode(result)
    sims = cosine_similarity([query_vector], doc_vectors)[0]
    max_index = np.argmax(sims)
    if doc_titles[max_index] != result:
        print(f"Mismatch found: {doc_titles[max_index]} vs {result}")
        mismatched_titles.append(linked_log)
        continue 
    result=result+'\n'+doc_texts[max_index]
    inputs.append(f'''Please determine whether the following text is related to this error message. 
        The error message is {error_log}.\n The text to be checked is here. {result}\n Return ‘Yes’ if they are related, or ‘No’ if they are not.''')
    outputs.append(random.choice(["Yes.", "Yes, they are related.", "Yes. It is related to the error message."]))
    min_index = np.argmin(sims)
    # most unrelated docs
    #print(f"most unrelated doc: {doc_titles[min_index]}")
    result=doc_titles[min_index]+'\n'+doc_texts[min_index]
    inputs.append(f'''Please determine whether the following text is related to this error message. 
        The error message is {error_log}.\n The text to be checked is here. {result}\n Return ‘Yes’ if they are related, or ‘No’ if they are not.''')
    outputs.append(random.choice(["No.", "No, they are not related.", "No. It is not related to the error message."]))
for linked_log in mismatched_titles:
    # remove them from data
    data.remove(linked_log)
with open(data_file, 'w') as f:
    json.dump(data, f, indent=4)
assert(len(inputs) == len(outputs)), "Inputs and outputs must have the same length"
print(f"Total inputs: {len(inputs)}")
data_dict = {
    "input": inputs,
    "output": outputs
}
dataset = Dataset.from_dict(data_dict)

'''model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_4bit=True,
    device_map="auto"
)'''

model = load_clm(model_name,
    load_in_4bit=True, device_map="auto",
    trust_remote_code=True)
if len(tokenizer) > model.get_input_embeddings().num_embeddings:
    model.resize_token_embeddings(len(tokenizer))

# PEFT (LoRA) 
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)
model = get_peft_model(model, peft_config)
dataset = dataset.map(lambda sample: {"prompt": sample["input"], "completion": sample["output"]})
dataset = dataset.map(tokenize)
#tokenized_dataset = dataset.map(tokenize_function, batched=True)

training_args = TrainingArguments(
    output_dir=model_save_path+"judge-llm",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=1e-5,
    num_train_epochs=1,
    logging_steps=10,
    logging_dir=model_save_path+"/logs",
    save_strategy="epoch",
    fp16=True,                                   # A6000은 fp16 빠름 (아니면 bf16도 가능)
    bf16=False,
    optim="paged_adamw_8bit"
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    #processing_class=tokenizer
)

trainer.train()
trainer.save_model(model_save_path+"judge-llm")       
tokenizer.save_pretrained(model_save_path+"judge-llm")  

# How to use:
peft_path = model_save_path+"judge-llm"
peft_config = PeftConfig.from_pretrained(peft_path)

# base 모델 불러오기 (4bit 동일 설정 필요)
base_model = AutoModelForCausalLM.from_pretrained(
    peft_config.base_model_name_or_path,
    load_in_4bit=True,
    device_map="auto"
)

# LoRA 적용
model = PeftModel.from_pretrained(base_model, peft_path)
for name, param in model.named_parameters():
    if torch.isnan(param).any():
        print(f"{name} contains NaNs!!!!!!!!")
    if torch.isinf(param).any():
        print(f"{name} contains Infs!!!!!!!!!!!!!!!!!!!!!")
# tokenizer 로딩
tokenizer = AutoTokenizer.from_pretrained(peft_path)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model.resize_token_embeddings(len(tokenizer))
prompt = "### Instruction:\n바나나는 어떤 과일인가요?\n\n### Response:\n"
inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(model.device)
outputs = model.generate(
    **inputs,
    max_new_tokens=100,
    do_sample=True,
    temperature=0.7,
    top_k=50,
    top_p=0.95,
    eos_token_id=tokenizer.eos_token_id
)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
