from unsloth import FastLanguageModel
# From 26.03.24, I will move to use unsloth to use less memory.

from curses import raw
import argparse
import os
os.environ["TRANSFORMERS_CACHE"]        = "/storage1/hf_cache/transformers"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"          # 선택
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import logging
import gc
import re
import copy
import math
from datetime import datetime
import json

from RAG import RAG
from kubernetes_config import *
from kubernetes import client, config
from typing import List, Dict, Optional

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, PeftConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
import torch
from trl import (
    GRPOTrainer, GRPOConfig,
    DPOTrainer,
    DPOConfig
)

from huggingface_hub import hf_hub_download
from llama_cpp import Llama

from datasets import Dataset
from torch.utils.tensorboard import SummaryWriter

#Local modules
from prompt import namespace

#from generate_pair_DPO import generate_io_pairs


logging.getLogger("paramiko").setLevel(logging.CRITICAL) 
from generate_pair_DPO import generate_io_pairs, _normalize_dpo_rows, _print_category_stats, parse_reasoning_and_final, filter_rows_by_token_length
from RL_config import *

def shape_reward(r):
    """Reward shaping: [0,1] → [-1,1]"""
    return 2 * r - 1

def normalize_rewards(reward_tensor):
    """Batch 수준 normalization"""
    mean = reward_tensor.mean()
    std = reward_tensor.std(unbiased=False)
    return (reward_tensor - mean) / (std + 1e-6)

def render_chat_prompt(tokenizer, user_text: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": user_text},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

def select_request_message(test_result: int, form: str, vnf: str, server_or_message=None, collection=None, embed_model=None) -> tuple[str, str, str]:
    # RAG setting
    rag_threshold = 0.8
    
    request_message = ''
    inform_message = ''
    error_message = ''
    if test_result == 1:
        request_message = f"I can't find {form} code in your response. Please modify it."
    elif test_result == 10:
        request_message = f"I failed to parse your {form} code. Please check the format and modify it."
    elif test_result == 11:
        server_or_message= "Your YAML code runs on localhost or Kubernetes nodes, but there's no task to perform there. Please update it to run only inside Kubernetes."
    elif test_result == 12:
        inform_message = "I found infinite loop in your code. "
        request_message = "Please modify your code to avoid infinite loop."
    elif test_result == 14:
        inform_message = "I can not find 'create_pod' function in your code. "
        request_message = 'Please check and modify your code.'
        if server_or_message:
            error_message = server_or_message
    elif test_result == 20: #RAG
        inform_message = 'While running your code, Pod error occured. Here are the logs from pods.\n'
        error_message = server_or_message
    elif test_result == 21: 
        inform_message = f'While running your code, Pod fell into {server_or_message} phase.\n'
    elif test_result == 22:
        inform_message = 'While running your code in my local machine, I got this exception.\n'+ str(server_or_message)
    elif test_result == 23:#RAG
        status, ansible_output = server_or_message
        inform_message = 'While running your code, Ansible runner result is ' + status + '.\n'
        error_message = str(ansible_output)
    elif test_result == 31:
        request_message = "It doesn't seem to be set to variable for 'pod_name' and 'namespace' in the created pod. Modify it to set to variable."
    elif test_result == 41:
        request_message = 'There was a task that was skipped due to an incorrect host name. Please correct the host name to use variable.'
    elif test_result == 43:
        print('43 error occured. somthing wrong?')
        inform_message = 'I got exception while finding pod in Kubernetes.'
        request_message = 'Please check the pod name and namespace are using variable, and correct it.'
    elif test_result == 51:#RAG
        inform_message = "When I run your code, 'create_pod' function returned False. And here are stdout from the function.\n"
        request_message = 'Please check your code and modify it run well.'
        error_message = server_or_message
    elif test_result == 90:#RAG
        inform_message = 'While running your code, Ansible runner takes too long time to run. It seems that your code is not running well.'
        error_message = server_or_message
        request_message = 'Please check your code and modify it to run well.'
    elif test_result == 91:
        inform_message = "While running your code, container didn't get ready for a long time. It seems that your code is not running well."
        request_message = 'Please check your code and modify it to run well.'
    elif test_result == 2:
        inform_message = server_or_message
    elif test_result == 31:
        inform_message = "It doesn't seem to be set to variable for 'pod_name' and 'namespace' in the created pod. Modify it to set to variable."
    elif test_result == 32:
        if server_or_message:
            inform_message = f"After I run your code, the container '{server_or_message}' exited."
        else:
            inform_message = "After I run your code, the container exited."                             
        request_message = "Please modify your code by adding 'sleep infinity' so that the container does not turn off. Please show me the updated version.\n"                        
    elif is_k8s_config_error_code(test_result):
        inform_message = 'While configure VNF with your code, I got this error. \n'
        error_message = server_or_message
    elif test_result == 33:
        container, error_code, error_message = server_or_message
        inform_message = f"An error with error code {error_code} occurred in the container '{container}' while operating your code. It means "

    if test_result not in [2, 31, 32]:
        retrieved_docs=RAG.RAG_search(error_message, collection, embed_model, vnf_name=vnf, use_tf_idf = False)
        retrieved_texts=''
        retrieved_well = False
        for retrieved_doc in retrieved_docs:
            if retrieved_doc['distance'] <rag_threshold:
                retrieved_texts += retrieved_doc['text']+'\n'
                retrieved_well = True
        if retrieved_well:
            request_message += '\nHere are some documents that may help you to modify your code:\n'+ retrieved_texts
    if request_message =='':
        request_message='Please correct your code and return the updated version by refering MOP again to configure VNF correctly.\n'
    return inform_message, request_message, error_message

def _load_eval_model_hf(model_path_or_id: str):
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    adapter_config_path = os.path.join(model_path_or_id, "adapter_config.json")
    tokenizer_source = model_path_or_id

    if os.path.exists(adapter_config_path):
        peft_config = PeftConfig.from_pretrained(model_path_or_id)
        base_model_id = peft_config.base_model_name_or_path
        if not os.path.exists(os.path.join(model_path_or_id, "tokenizer_config.json")):
            tokenizer_source = base_model_id
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            torch_dtype=torch.bfloat16,
            quantization_config=quantization_config,
            device_map="auto",
            cache_dir=hf_cache_path,
        )
        model = PeftModel.from_pretrained(base_model, model_path_or_id, is_trainable=False)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path_or_id,
            torch_dtype=torch.bfloat16,
            quantization_config=quantization_config,
            device_map="auto",
            cache_dir=hf_cache_path,
        )

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, cache_dir=hf_cache_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model.eval()
    if hasattr(model, "config"):
        model.config.use_cache = True
    return model, tokenizer

def get_response_from_llm_unsloth(model, tokenizer, max_new_tokens, temperature, top_p, prompt, do_sample=False) -> str:
    enc = build_enc(tokenizer, prompt, max_input_tokens=8192)
    input_ids = enc["input_ids"]
    if hasattr(input_ids, "input_ids"):
        input_ids = input_ids["input_ids"]
    first_device = next(model.parameters()).device
    input_ids = input_ids.to(first_device)
    attention_mask = torch.ones_like(input_ids)

    def _walk_model_chain(model_obj):
        modules = []
        current = model_obj
        while True:
            modules.append(current)
            if not hasattr(current, "model"):
                break
            current = current.model
        return modules

    def _clear_generation_flags(model_obj):
        saved = []
        for module in _walk_model_chain(model_obj):
            had_flag = hasattr(module, "_flag_for_generation")
            saved.append((module, had_flag))
            if had_flag:
                del module._flag_for_generation
        return saved

    generation_kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.eos_token_id,
        "repetition_penalty": 1.1,
        "eos_token_id": tokenizer.eos_token_id,
        "use_cache": False,
    }
    if do_sample:
        generation_kwargs.update({
            "temperature": temperature,
            "top_p": top_p,
            "top_k": 50,
        })

    generate_fn = model.generate
    if hasattr(model, "base_model") and hasattr(model.base_model, "_old_generate"):
        generate_fn = model.base_model._old_generate
    elif hasattr(model, "_old_generate"):
        generate_fn = model._old_generate

    old_use_cache = getattr(model.config, "use_cache", None) if hasattr(model, "config") else None
    saved_generation_flags = _clear_generation_flags(model)
    model.eval()
    if hasattr(model, "config"):
        model.config.use_cache = False

    try:
        with torch.no_grad():
            out_ids = generate_fn(**generation_kwargs)
    finally:
        if hasattr(model, "config") and old_use_cache is not None:
            model.config.use_cache = old_use_cache
        for module, had_flag in saved_generation_flags:
            if had_flag:
                module._flag_for_generation = True

    input_len = input_ids.shape[1]
    gen_ids = out_ids[0][input_len:]
    answer = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    return answer

def evaluate_llm(
    model_path_or_id: str,
    mop_data: List[Dict[str, str]],
    k8s_client, # Kubernetes client
    form : str = 'Python',
    retry: bool = False,
    max_prompt_length: int = 6500,
    max_new_tokens: int = 128,
    temperature: float = 0.3,
    top_p: float = 0.8,
    device: Optional[str] = None,
    using_cot: bool = False,
    using_gguf: bool = False,
    ):
    use_whole_code = True if using_cot else False
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    gc.collect()
    torch.cuda.empty_cache()
    if using_gguf:
        filename = "qwen3.5-9b-claude-4.6-opus-reasoning-distilled.Q4_K_M.gguf"
        model_path_or_id = hf_hub_download(repo_id=model_path_or_id,filename=filename, cache_dir=hf_cache_path)
        model = Llama(model_path_or_id, n_ctx=max_prompt_length, n_gpu_layers=-1, temperature=temperature, top_p=top_p, max_tokens=max_new_tokens, verbose=False)
        messages = [
            {"role": "system", "content": "You are a Kubernetes and system automation expert. Provide accurate MOP logic."},
            {"role": "user", "content": ""}
        ]
    else:
        if retry:
            #RAG initiation
            db_list=['RAG/stackoverflow_docs.db', 'RAG/kubernetes_docs.db']
            if form == 'Ansible':
                db_list.append('RAG/ansible_docs.db')
            collection, embed_model = RAG.RAG_init(db_list, embed_model='fine-tuned', new=True)
        max_seq_length = max_prompt_length + max_new_tokens
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path_or_id,
            max_seq_length=max_seq_length,
            dtype=torch.bfloat16,
            load_in_4bit=True,
            cache_dir=hf_cache_path,
        )
        # decoder-only 모델의 padding 문제 방지
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        FastLanguageModel.for_inference(model)
        eval_backend = "unsloth"

    vm_num = {}
    # 배치로 처리하고 싶으면 batch_size를 추가로 받아서 chunking하면 됨.
    success_num=0
    model_name = model_path_or_id.split('/')[-1]
    for single_mop_data in mop_data:
        prompt = single_mop_data['mop']
        vnf = single_mop_data['vnf']
        if vnf not in vm_num:
            vm_num[vnf] = 1
        else:
            vm_num[vnf] += 1
        if using_gguf:
            messages[1]['content'] = prompt
            response = model.chat(messages, max_tokens=max_new_tokens, temperature=temperature, top_p=top_p)
            answer = response['choices'][0]['message']['content']
        else:
            try:
                if eval_backend == "unsloth":
                    answer = get_response_from_llm_unsloth(model, tokenizer, max_new_tokens, temperature, top_p, prompt)
                else:
                    answer = get_response_from_llm(model, tokenizer, max_new_tokens, temperature, top_p, prompt)
            except RuntimeError as exc:
                if eval_backend == "unsloth" and "doesn't match the broadcast shape" in str(exc):
                    print("[WARN] Unsloth evaluation generate failed on a long prompt; switching to HF generation fallback.")
                    del model
                    gc.collect()
                    torch.cuda.empty_cache()
                    model, tokenizer = _load_eval_model_hf(model_path_or_id)
                    eval_backend = "hf"
                    answer = get_response_from_llm(model, tokenizer, max_new_tokens, temperature, top_p, prompt)
                else:
                    raise
        if using_cot:
            answer = parse_reasoning_and_final(answer)[1]
        success, test_result = get_reward_from_llm_response(answer, form, vnf, model_name, vm_num, k8s_client, namespace, use_whole_code=use_whole_code)
        if success == 1.0:
            success_num+=1
        elif retry: #Retry once, use RAG if need.
            status_code, server_or_message = test_result
            inform_message, request_message, error_message =  select_request_message(status_code, form, vnf, server_or_message, collection, embed_model)
            prompt = prompt+answer+inform_message+error_message+request_message
            if using_gguf:
                messages[1]['content'] = prompt
                response = model.chat(messages, max_tokens=max_new_tokens, temperature=temperature, top_p=top_p)
                answer = response['choices'][0]['message']['content']
            else:
                if eval_backend == "unsloth":
                    answer = get_response_from_llm_unsloth(model, tokenizer, max_new_tokens, temperature, top_p, prompt)
                else:
                    answer = get_response_from_llm(model, tokenizer, max_new_tokens, temperature, top_p, prompt)
            if using_cot:
                answer = parse_reasoning_and_final(answer)[1]
            success, test_result = get_reward_from_llm_response(answer, form, vnf, model_name, vm_num, k8s_client, namespace, use_whole_code=use_whole_code)
            if success == 1.0:
                success_num+=1

    print (f'[INFO] Evaluation completed. Success rate: {success_num}/{len(mop_data)}')
    return success_num/len(mop_data)

def train_grpo(
    model_path_or_id: str,
    mop_data: List[Dict[str, str]],
    k8s_client,  # Kubernetes client
    form: str = "Python",
    output_dir: str = "./tmp/grpo-out",
    learning_rate: float = 1e-5,
    beta_kl: float = 0.005,
    max_prompt_length = 6500,
    max_new_tokens: int = 500,
    temperature: float = 0.7,
    top_p: float = 0.9,
    steps: int = 80,
    batch_size: int = 2,
    grad_accum: int = 8,
    num_generations: int = 8,
    num_prompts_per_step: int = 4,
    scale_rewards: str = "group",
    log_completions: bool = False,
    num_completions_to_print: int = 0,
    using_cot:bool =False,
    ):
    
    print(f"[INFO] Starting GRPO training with Unsloth Ultra-reduced memory settings 🚀")
    gc.collect()
    torch.cuda.empty_cache()
    print(f"[INFO] GPU memory before model load: {torch.cuda.memory_allocated()/1024**3:.2f} GB allocated, {torch.cuda.memory_reserved()/1024**3:.2f} GB reserved.")
    output_dir_logs = os.path.join(output_dir, "logs_"+datetime.now().strftime('%m%d_%H%M'))

    use_whole_code = True if using_cot else False

    # -------------------------------
    # 1) 모델/토크나이저 로드 (🔥 Unsloth 적용)
    # -------------------------------
    max_seq_length = max_prompt_length + max_new_tokens

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_path_or_id,
        max_seq_length = max_seq_length,
        dtype = torch.bfloat16,
        load_in_4bit = True,
    )

    # PEFT(LoRA) 적용
    model = FastLanguageModel.get_peft_model(
        model,
        r = 16,
        lora_alpha = 32,
        # 기존 모듈에 MLP 레이어 추가 권장
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", 
                          "gate_proj", "up_proj", "down_proj"],
        lora_dropout = 0, # 최적화를 위해 0 강제
        bias = "none",
        use_gradient_checkpointing = "unsloth", # ⚠️ 메모리 절약의 핵심! (경고 없이 완벽 작동)
        random_state = 3407,
    )

    # ❌ (삭제됨) prepare_model_for_kbit_training 및 requires_grad 루프, 캐스팅 루프 전부 불필요
    
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left" 

    # -------------------------------
    # 2) Dataset 설정 (기존 유지)
    # -------------------------------
    train_rows = [{
        "prompt": render_chat_prompt(tokenizer, x["mop"]),
        "vnf": x["vnf"],
    } for x in mop_data]
    train_dataset = Dataset.from_list(train_rows)
    writer = SummaryWriter(output_dir_logs)
    _cache = {"step": None, "raw": [], "shaped": []}

    model_name = model_path_or_id.split("/")[-1] + "_GRPO"

    # -------------------------------
    # 3) Custom Reward 함수 (기존 유지)
    # -------------------------------
    def reward_func(prompts, completions, vnf, trainer_state=None, **kwargs):
        raw_rewards = []
        shaped = []
        if len(vnf) != len(completions):
            # print("## Adjusting vnf/prompts to match completions length")
            reps = len(completions) // len(vnf)
            vnf = [v for v in vnf for _ in range(reps)]
            prompts = [p for p in prompts for _ in range(reps)]
            
        for v, comp, prompt in zip(vnf, completions, prompts):
            if using_cot:
                comp = parse_reasoning_and_final(comp)[1]
            # namespace 변수는 외부 스코프에 정의되어 있다고 가정합니다.
            raw, _ = get_reward_from_llm_response(comp, form, v, model_name, {v:1}, k8s_client, namespace, logging=True, use_whole_code=use_whole_code)
            raw_rewards.append(float(raw))
            shaped.append(5.0*float(shape_reward(raw)))

        shaped_t = torch.tensor(shaped, dtype=torch.float32)

        if trainer_state is not None:
            step = int(trainer_state.global_step)
            if _cache["step"] is not None and step != _cache["step"]:
                if _cache["raw"]:                    
                    writer.add_scalar("custom_reward/raw_mean", sum(_cache["raw"]) / len(_cache["raw"]), _cache["step"])
                    writer.add_scalar("custom_reward/shaped_mean", sum(_cache["shaped"]) / len(_cache["shaped"]), _cache["step"])
                _cache["raw"].clear()
                _cache["shaped"].clear()

            _cache["step"] = step
            _cache["raw"].extend(raw_rewards)
            _cache["shaped"].extend(shaped)
        assert all(math.isfinite(r) for r in shaped_t)

        return [float(x) for x in shaped_t.detach().cpu().tolist()]

    # -------------------------------
    # 4) GRPOConfig (수정됨)
    # -------------------------------
    args = GRPOConfig(
        output_dir=output_dir,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        max_steps=steps,
        logging_steps=1,
        report_to=["tensorboard"],
        logging_dir=output_dir_logs,

        max_prompt_length=max_prompt_length, 
        generation_batch_size=num_generations*num_prompts_per_step,
        num_generations=num_generations,
        max_completion_length=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        beta=beta_kl,
        scale_rewards=scale_rewards,
        bf16=True,
        fp16=False,
        
        # ⚠️ Unsloth를 사용하면 아래 항목을 켜도 Warning이 발생하지 않습니다!
        gradient_checkpointing=True, 
        
        log_completions=log_completions,
        num_completions_to_print=num_completions_to_print,
    )

    # -------------------------------
    # 5) Trainer 구성 & 학습
    # -------------------------------
    trainer = GRPOTrainer(
        model=model,
        args=args,
        reward_funcs=reward_func,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )
    
    with torch.autocast("cuda"):
        trainer.train()

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"\n✅ GRPO training finished. Saved to: {output_dir}")

###############################################
# 2. ONLINE DPO TRAINING LOOP
###############################################
def train_dpo(
    model_path_or_id: str,
    dpo_rows: List[Dict[str, str]],
    output_dir: str = "./tmp/dpo-out",
    per_device_train_batch_size: int = 2,
    gradient_accumulation_steps: int = 1,
    learning_rate: float = 2e-6,
    max_prompt_length: int = 6500,
    max_new_tokens: int = 500,
    num_train_epochs: int = 1,
    beta: float = 0.03,
    device: Optional[str] = None,
    logging_steps: int = 1,
    using_cot: bool = False,
    test: Optional[int] = None,
):
    """
    dpo_rows(list of dict)로 Dataset 만든 뒤 TRL의 DPOTrainer로 학습합니다. (Unsloth 최적화 적용)
    """
    gc.collect()
    torch.cuda.empty_cache()
    output_dir_logs = os.path.join(output_dir, "logs_"+datetime.now().strftime('%m%d_%H%M'))
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # -------------------------------
    # 1) Context 길이 선언
    # -------------------------------
    max_length = max_prompt_length+max_new_tokens

    print(
        f"[INFO] DPO memory budget: per_device_train_batch_size={per_device_train_batch_size}, "
        f"gradient_accumulation_steps={gradient_accumulation_steps}, "
        f"max_prompt_length={max_prompt_length}, max_length={max_length}"
    )

    # -------------------------------
    # 2) 모델/토크나이저 로드 (🔥 Unsloth 적용 부분)
    # -------------------------------
    print("🚀 Unsloth를 사용하여 모델을 로드합니다...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_path_or_id,
        max_seq_length = max_length,  # 위에서 결정된 max_length 사용
        dtype = torch.bfloat16,       # BF16 자동 적용
        load_in_4bit = True,          # 4-bit 양자화 로드 (QLoRA)
        cache_dir = hf_cache_path,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # -------------------------------
    # 3) Dataset 구성
    # -------------------------------
    # tokenizer가 로드된 이후에 Dataset 처리를 수행합니다.
    if test is not None:
        dpo_rows = random.sample(dpo_rows, int(len(dpo_rows)/test))
    if using_cot:
        dpo_rows = _normalize_dpo_rows(dpo_rows, tokenizer=tokenizer, shuffle=True, print_token_stats=True)
        dpo_rows, overflow_rows = filter_rows_by_token_length(
            dpo_rows,
            tokenizer=tokenizer,
            max_prompt_length=max_prompt_length,
            max_length=max_length,
            drop_overflow=True,
        )
        _print_category_stats(dpo_rows)
    
    train_dataset = Dataset.from_list(dpo_rows)

    # -------------------------------
    # 4) PEFT(LoRA) 적용 (🔥 Unsloth 적용 부분)
    # -------------------------------
    # prepare_model_for_kbit_training 등은 Unsloth가 내부적으로 처리하므로 삭제
    model = FastLanguageModel.get_peft_model(
        model,
        r = 16,
        lora_alpha = 32,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
        lora_dropout = 0, # ⚠️ Unsloth 최적화를 위해 무조건 0으로 설정
        bias = "none",
        use_gradient_checkpointing = "unsloth", # ⚠️ 메모리 절약의 핵심
        random_state = 3407,
    )
    model.print_trainable_parameters()

    # -------------------------------
    # 5) DPOConfig 설정 (기존 유지)
    # -------------------------------
    dpo_config = DPOConfig(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        beta=beta,
        bf16=True,
        fp16=False,
        remove_unused_columns=False,
        logging_steps=logging_steps,
        logging_dir=output_dir_logs,
        max_prompt_length=max_prompt_length,
        max_length=max_length,
        truncation_mode="keep_end",
        use_logits_to_keep=True,
        precompute_ref_log_probs=True,
        precompute_ref_batch_size=1,
        torch_empty_cache_steps=1,
        report_to='tensorboard',
        save_steps=200,
    )

    # -------------------------------
    # 6) Trainer 생성 및 학습
    # -------------------------------
    # Unsloth 모델은 TRL의 DPOTrainer와 완벽하게 호환됩니다.
    trainer = DPOTrainer(
        model=model,
        args=dpo_config,
        processing_class=tokenizer,   # tokenizer를 넘겨주면 내부에서 토큰화/패딩 처리
        train_dataset=train_dataset,
    )

    trainer.train()
    
    # 모델 저장
    trainer.save_model(output_dir)
    # tokenizer도 명시적으로 저장해 주는 것이 좋습니다.
    tokenizer.save_pretrained(output_dir) 
    
    return output_dir

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--Ansible', action='store_true')
    argparser.add_argument('--Python', action='store_true')
    argparser.add_argument('--RAG', action='store_true')
    argparser.add_argument('--model', default = 'qwen2.5-coder', choices = ['llama3.1', 'qwen2.5-coder', 'qwen3.5', 'gpt-oss', 'mistral', 'qwen3.5-claude4.6'])
    argparser.add_argument('--method', default = 'dpo', choices = ['dpo', 'ppo', 'grpo', 'dpo-ppo','dpo-grpo', 'evaluate-only'])
    argparser.add_argument('--load-data', type=str, default=None, help='Path to the DPO dataset to load')
    argparser.add_argument('--load-model', type=str, default=None, help='Path to the model to load')
    argparser.add_argument('--test',  type=int, help='Test with subsampled MOP data by a factor of N.')
    argparser.add_argument('--evaluate-first', action='store_true', help='Evaluate the base model before training')
    argparser.add_argument('--cot', action='store_true', help='Use chain-of-thought prompting')

    argparser=argparser.parse_args()
    #mop_file_path = '../mop/Intergrated/'
    mop_file_path = '../data_generating/data_v3/'
    system_name='Kubernetes'
    config.load_kube_config()
    v1 = client.CoreV1Api()
    apps_v1 = client.AppsV1Api()
    form='Python' if argparser.Python else 'Ansible'

    mop_data = read_mop_file(mop_file_path, system_name, test=argparser.test)

    model_list = {'llama3.1':"unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
                  'qwen2.5-coder':"unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit",
                  'qwen3.5':"unsloth/Qwen3.5-9B-Instruct-bnb-4bit",
                  'gpt-oss':"unsloth/gpt-oss-20b-bnb-4bit",
                  'mistral':"unsloth/Mistral-Nemo-Instruct-2407-bnb-4bit",
                  'qwen3.5-claude4.6': 'Jackrong/Qwen3.5-9B-Claude-4.6-Opus-Reasoning-Distilled-GGUF'}
    if argparser.load_model is not None:
        model_path_or_id=argparser.load_model
    else:
        model_path_or_id=model_list[argparser.model]
    original_success_rate = None
    max_prompt_length, max_new_tokens = 6500, 1024
    if argparser.evaluate_first:
        original_success_rate = evaluate_llm(
            model_path_or_id=model_path_or_id,
            mop_data=mop_data,
            retry=argparser.RAG,
            k8s_client=(v1,apps_v1),
            form=form,
            max_prompt_length=max_prompt_length,
            max_new_tokens=max_new_tokens,
            using_cot=argparser.cot,
        )
    if 'dpo' in argparser.method:
        # DPO 파라미터 자동 계산
        dpo_data_num = 200
        current_data_num = 0
        if argparser.cot:
            if argparser.load_data is None:
                print('CoT data set need to be loaded')
                exit(1)
            with open(argparser.load_data, 'r', encoding='utf-8') as f:
                input_io_pairs = json.load(f)
        else:
            if argparser.load_data is not None:
                import pickle
                with open(argparser.load_data, 'rb') as f:
                    input_io_pairs = pickle.load(f)
                current_data_num = len(input_io_pairs)
                print (f'Loaded {current_data_num} IO pairs from {argparser.load_data}')
            while current_data_num < dpo_data_num:
                from generate_pair_DPO import generate_io_pairs
                new_input_io_pairs = generate_io_pairs(
                    model_path_or_id=model_path_or_id,
                    mop_data=mop_data,
                    k8s_client=(v1,apps_v1),
                    form=form,
                    max_new_tokens=1000,
                )
                input_io_pairs = input_io_pairs + new_input_io_pairs if current_data_num > 0 else new_input_io_pairs
                current_data_num = len(input_io_pairs)
            input_io_pairs = input_io_pairs[:dpo_data_num]
            print (f'[INFO] Total {len(input_io_pairs)} IO pairs ready for DPO training.')
            with open('tmp/dpo_dataset.pkl' , 'wb') as f:
                import pickle
                pickle.dump(input_io_pairs, f)        
        # DPO 파라미터 동적 계산
        dpo_params = calculate_dynamic_params(
            data_size=len(input_io_pairs),
            training_type="dpo",
            using_cot=argparser.cot
        )
        max_prompt_length = dpo_params["max_prompt_length"]
        max_new_tokens = dpo_params["max_new_tokens"]
        train_dpo(
            model_path_or_id=model_path_or_id,
            dpo_rows=input_io_pairs,
            output_dir='./tmp/dpo_' + argparser.model + '_k8s',
            per_device_train_batch_size=dpo_params["batch_size"],
            gradient_accumulation_steps=dpo_params["grad_accum"],
            num_train_epochs=dpo_params["steps"],
            max_prompt_length = max_prompt_length,
            max_new_tokens = max_new_tokens,
            using_cot=argparser.cot,
            test= argparser.test,
        )
        # Memory cleanup after DPO training
        gc.collect()
        torch.cuda.empty_cache()
        print("[INFO] Memory cleaned up after DPO training.")
        print(f"[INFO] GPU memory after cleanup: {torch.cuda.memory_allocated()/1024**3:.2f} GB allocated, {torch.cuda.memory_reserved()/1024**3:.2f} GB reserved.")
        model_path_or_id='./tmp/dpo_' + argparser.model + '_k8s'
    
    if 'grpo' in argparser.method:
        print(f"[INFO] Starting GRPO training. GPU memory: {torch.cuda.memory_allocated()/1024**3:.2f} GB allocated, {torch.cuda.memory_reserved()/1024**3:.2f} GB reserved.")
        # GRPO 파라미터 동적 계산
        grpo_training_type = "grpo_trajectory" if argparser.traj else "grpo"
        grpo_params = calculate_dynamic_params(
            data_size=len(mop_data),
            training_type=grpo_training_type,
            using_cot=argparser.cot
        )
        max_prompt_length = grpo_params["max_prompt_length"]
        max_new_tokens = grpo_params["max_new_tokens"]
        
        if argparser.test:
            # 테스트 모드에서는 동적 계산 값을 무시하고 작은 값으로 설정
            grpo_params["steps"] = 2
            grpo_params["num_generations"] = 4
            grpo_params["num_prompts_per_step"] = 2
             
        else:
            train_grpo(
                model_path_or_id=model_path_or_id,
                mop_data=mop_data,
                k8s_client=(v1,apps_v1),
                form=form,
                output_dir='./tmp/grpo_' + argparser.model + '_k8s',
                steps=grpo_params["steps"],
                num_generations=grpo_params["num_generations"],
                num_prompts_per_step=grpo_params["num_prompts_per_step"],
                grad_accum=grpo_params["grad_accum"],
                max_prompt_length=max_prompt_length,
                max_new_tokens=max_new_tokens,
                using_cot=argparser.cot,
            )
            model_path_or_id='./tmp/grpo_' + argparser.model + '_k8s'
    new_success_rate = evaluate_llm(
        model_path_or_id=model_path_or_id,
        mop_data=mop_data,
        retry=argparser.RAG,
        k8s_client=(v1,apps_v1),
        form=form,
        max_prompt_length=max_prompt_length,
        max_new_tokens=max_new_tokens,
        using_cot=argparser.cot,
    )
    if original_success_rate is not None:
        print (f'[INFO] Success rate improved from {original_success_rate:.2%} to {new_success_rate:.2%} after training.')