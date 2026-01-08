from already_done import already_done
#from langchain.chat_models import ChatOpenAI
import argparse

from docx import Document
import os
os.environ["TRANSFORMERS_CACHE"]        = "/storage1/hf_cache/transformers"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"          # 선택
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import logging
import time
import random
import gc

#from RAG import RAG
from kubernetes_config import *
from kubernetes import client, config
from typing import List, Dict, Optional

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from trl import (
    GRPOTrainer, GRPOConfig,
    PPOConfig,
    PPOTrainer,
    AutoModelForCausalLMWithValueHead,
    create_reference_model,
    DPOTrainer,
    DPOConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset
from torch.utils.tensorboard import SummaryWriter

#Local modules
from prompt import prompt, namespace
prompts_1, prompts_2, example_data = prompt('Python', 'Kubernetes') # Prompots for each language and system.
good_example, good_example_prefix, goood_example_suffix = example_data
from openstack_config import * 
from kubernetes_config import *


logging.getLogger("paramiko").setLevel(logging.CRITICAL) 
hf_cache_path = '/storage3/hf_cache/'
#bnb_config = BitsAndBytesConfig(load_in_8bit=True)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

def build_enc(tokenizer, prompt_text: str, max_input_tokens: int = 8192):
    messages = [{"role": "user", "content": prompt_text}]
    # add_generation_prompt=True 가 핵심: "이제 모델이 답할 차례"를 붙여줌
    enc = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        truncation=True,
        return_tensors="pt",
        max_length=max_input_tokens,
    )
    return {"input_ids": enc}

def shape_reward(r):
    """Reward shaping: [0,1] → [-1,1]"""
    return 2 * r - 1

def normalize_rewards(reward_tensor):
    """Batch 수준 normalization"""
    mean = reward_tensor.mean()
    std = reward_tensor.std(unbiased=False)
    return (reward_tensor - mean) / (std + 1e-6)

def read_mop_file(file_path: str, test:bool = False) -> list[str]:
    mop_list = [file_name for file_name in os.listdir(file_path)]

    # Just be simple
    if test:
        mop_len = len(mop_list)
        mop_list = mop_list[:int(mop_len/8)]  # Test with 1/8 MOPs

    mop_data = []
    for mop_file in mop_list:
        vnf = mop_file.split('_')[0]
        lang = mop_file.split('_')[3]
        action = mop_file.split('_')[2]
        if action == 'port':
            additional_action='Based on it, configure to block the port except 22 and 80.\n'
        elif action in ['subnet', 'block'] :
            additional_action='Based on it, configure to block the traffic except subnets with'
            if system_name == 'OpenStack':
                additional_action+=' 10.10.10.0/24 and 10.10.20.0/24.\n'
            else:
                additional_action+=' 10.244.0.0/16.\n'
            if vnf=='nDPI':
                additional_action+= "Configure nDPI to block the traffic, don't use the firewall.\n"
        else:
            additional_action=''
        if vnf == 'Suricata':
            additional_action+="To install Suricata, you may need to add apt repository of Suricata, so you may need to install 'software-properties-common' first, and then add Suricata repository.\n"
        
        example_code_list = [example + ':\n'+ good_example[example] for example in good_example if not vnf.lower() in example.lower()] 
        example_code = '\n'.join(random.sample(example_code_list, min(len(example_code_list), 2))) #Select 2 examples randomly
        example_code = good_example_prefix + example_code + goood_example_suffix

        # Parameters for VM creation
        vm_name = 'vm-'+vnf+'-'+str(1)
        mop=''
        assert (mop_file.endswith('.docx'))
        doc = Document(mop_file_path + mop_file)
        for para in doc.paragraphs:
            mop += para.text + '\n'
        single_mop = {'vnf': vnf, 'lang': lang, 'mop': prompts_1+prompts_2+ mop +additional_action+ example_code, 'vm_name': vm_name}
        mop_data.append(single_mop)
    return mop_data        

def get_response_from_llm(model, tokenizer, enc, max_new_tokens, temperature, top_p, prompt, do_sample=False) -> str:
    #assert tokenizer.name_or_path == model.config.name_or_path
    attention_mask = torch.ones_like(enc["input_ids"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with torch.no_grad():
        if do_sample:        
            out_ids = model.generate(
                input_ids=enc["input_ids"].to(device),
                attention_mask=attention_mask.to(device),
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=tokenizer.eos_token_id,
                top_k = 50,
                repetition_penalty=1.1,
                eos_token_id=tokenizer.eos_token_id
            )
        else:        
            out_ids = model.generate(
                input_ids=enc["input_ids"].to(device),
                attention_mask=attention_mask.to(device),
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1,
                eos_token_id=tokenizer.eos_token_id
            )

    # 보통 decoded에는 prompt+answer가 같이 들어있으므로,
    # prompt를 제거해서 "answer만" 분리해두는게 실용적입니다.

    input_len = enc["input_ids"].shape[1]
    output_len = out_ids.shape[1]
    print("\n##>> input_len:", input_len, "output_len:", output_len, "new_tokens:", output_len - input_len)
    gen_ids = out_ids[0][input_len:]
    # 디코딩
    answer = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    return answer

def get_reward_from_llm_response(answer: str, form: str, vnf: str, model_name: str, vm_num: int, v1, namespace: str) -> float:
    if form == 'Python':
        test_result, server_or_message = test_creation_python_K8S(answer, vnf, model_name, vm_num, 1, v1, namespace)
    else:
        test_result, server_or_message = test_creation_ansible_K8S(answer, vnf, model_name, vm_num, v1, 600)
    if test_result == 0: # Creation Success
        second_test_result, second_message = test_K8S_configuration(server_or_message, vnf, v1, namespace)
        delete_pod(v1, server_or_message, namespace)
        print(f'## Creation success, status code: {second_test_result}')
        if second_test_result == 0:
            # Both Creation and Configuration Success
            reward = 1.0
        # Now, make reward based on status code.
        elif second_test_result == 1:
            reward = 0.0
        elif second_test_result == 2:
            reward = 0.8
        elif second_test_result in [10, 11, 12, 13, 14]:
            reward = 0.2
        elif second_test_result in [20, 21, 22, 23]:
            reward = 0.4
        elif second_test_result in [30, 31, 32, 33, 41, 43, 51, 90, 91]:
            reward = 0.6
        else:
            print( '## Unknown error code from configuration test:', second_test_result)
            return None
    else:
        print('## Creation failed')
        # VM Creation failed
        reward = 0.0
    return reward
###############################################
# Generate IO Pairs for DPO Training
# Input data for DPO should be in the form of
# {"prompt": ..., "chosen": ..., "rejected": ...}
###############################################
def generate_io_pairs(
    model_path_or_id: str,
    mop_data: List[Dict[str, str]],
    v1, # Kubernetes client
    form : str = 'Python',
    max_new_tokens: int = 500,
    temperature: float = 0.3,
    top_p: float = 0.8,
    device: Optional[str] = None,
) -> List[Dict[str, str]]:
    """
    MOP data (list[dict[str, str]])를 받아 LLM 출력 생성 후,
    [{"input": ..., "chosen": ..., "rejected": ...}, ...] 형태로 반환.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    gc.collect()
    torch.cuda.empty_cache()
    tokenizer = AutoTokenizer.from_pretrained(model_path_or_id, cache_dir=hf_cache_path)
    # decoder-only 모델의 padding 문제 방지
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_path_or_id, quantization_config=bnb_config,device_map="auto",cache_dir=hf_cache_path)
    model.eval()

    results: List[Dict[str, str]] = []
    vm_num = {}
    # 배치로 처리하고 싶으면 batch_size를 추가로 받아서 chunking하면 됨.
    model_name = model_path_or_id.split('/')[-1]
    for single_mop_data in mop_data:
        prompt = single_mop_data['mop']
        vnf = single_mop_data['vnf']
        if vnf not in vm_num:
            vm_num[vnf] = 1
        else:
            vm_num[vnf] += 1
        lang = single_mop_data['lang']
        vm_name = single_mop_data['vm_name']
        enc = build_enc(tokenizer, prompt, max_input_tokens=8192)
        first_device = next(model.parameters()).device
        enc = {k: v.to(first_device) for k, v in enc.items()}
        print ('\n'+'#'*50)
        print (f'## Generating IO pairs for VNF: {vnf}, Language: {lang}, VM: {vm_name}, MOP:')
        #print(prompt)
        #print('\n ## Now here is answer')
        answer = get_response_from_llm(model, tokenizer, enc, max_new_tokens, temperature, top_p, prompt, do_sample=True)
        #print(answer)

        # Test in NDT
        success = get_reward_from_llm_response(answer, form, vnf, model_name, vm_num, v1, namespace)
        if success == 1.0:
            good_config_answer = answer
            print('## Good configuration example generated successfully.')
            for _ in range(5):  # 최대 5번까지 시도
                answer = get_response_from_llm(model, tokenizer, enc, max_new_tokens, min(0.7, temperature+0.1), top_p, prompt, do_sample=True)
                success = get_reward_from_llm_response(answer, form, vnf, model_name, vm_num, v1, namespace)
                if success != 1.0:
                    bad_config_answer = answer
                    break
            print('## Bad configuration example generated successfully, added to IO pairs.')
            results.append({"input": prompt, "chosen": good_config_answer, "rejected": bad_config_answer})
    print (f'Total {len(results)} IO pairs generated for DPO training.')
    return results


def evaluate_llm(
    model_path_or_id: str,
    mop_data: List[Dict[str, str]],
    v1, # Kubernetes client
    form : str = 'Python',
    max_new_tokens: int = 128,
    temperature: float = 0.3,
    top_p: float = 0.8,
    device: Optional[str] = None,
    ):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    gc.collect()
    torch.cuda.empty_cache()
    tokenizer = AutoTokenizer.from_pretrained(model_path_or_id, cache_dir=hf_cache_path)
    # decoder-only 모델의 padding 문제 방지
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_path_or_id, quantization_config=bnb_config,device_map="auto",cache_dir=hf_cache_path)
    model.eval()

    results: List[Dict[str, str]] = []
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
        lang = single_mop_data['lang']
        vm_name = single_mop_data['vm_name']
        enc = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            padding=False,
        ).to(device)
        answer = get_response_from_llm(model, tokenizer, enc, max_new_tokens, temperature, top_p, prompt)
        success = get_reward_from_llm_response(answer, form, vnf, model_name, vm_num, v1, namespace)
        if success == 1.0:
            success_num+=1
    print (f'## Evaluation completed. Success rate: {success_num}/{len(mop_data)}')
    return success_num/len(mop_data)

###############################################
# 1. PPO TRAINING LOOP
###############################################


## PPO onling training doesn't work in latest TRL version (0.11~)
# Stop now.
def train_ppo(
    model_path_or_id: str,
    mop_data: List[Dict[str, str]],
    v1, # Kubernetes client
    form : str = 'Python',
    output_dir: str = "./tmp/ppo-out",
    learning_rate: float = 1e-6,
    target_kl: float = 0.1,
    max_new_tokens: int = 500,
    temperature: float = 0.2,
    top_p: float = 0.8,
    device: Optional[str] = None,
    steps: int = 20,
    ):
    print("\n=== RUNNING PPO TRAINING ===")
    gc.collect()
    torch.cuda.empty_cache()
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = AutoModelForCausalLMWithValueHead.from_pretrained(model_path_or_id, cache_dir=hf_cache_path, 
                                                              quantization_config=bnb_config,device_map="auto")
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    tokenizer = AutoTokenizer.from_pretrained(model_path_or_id, cache_dir=hf_cache_path)
    tokenizer.pad_token = tokenizer.eos_token

    ref_model = create_reference_model(model)

    # PPO Config with KL control + reward normalization
    config = PPOConfig(
        output_dir=output_dir,
        learning_rate=learning_rate,
        batch_size=3,
        mini_batch_size=1,
        gradient_accumulation_steps=4,
        kl_coef=target_kl,

        report_to="tensorboard",
        logging_dir=output_dir+'/logs',

    )
    trainer = PPOTrainer(
        args=config,
        model=model,
        ref_model=ref_model,
        processing_class=tokenizer,
    )
    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": True,
        "top_p": top_p,
        "temperature": temperature,
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    writer = SummaryWriter(output_dir+'/logs')

    model_name = model_path_or_id.split('/')[-1]+'_PPO'
    for step in range(steps):
        batch_samples = random.sample(mop_data, config.batch_size)
        batch_prompts = [sample['mop'] for sample in batch_samples]
        vnf_list = [sample['vnf'] for sample in batch_samples]
        inputs = tokenizer(batch_prompts, return_tensors="pt").to(device)

        responses_ids = trainer.generate(inputs["input_ids"], **generation_kwargs)
        responses_list = []
        response_ids_list = []
        for i in range(config.batch_size):
            input_len = inputs["input_ids"][i].shape[0]
            gen_ids = responses_ids[i][input_len:]
            response_ids_list.append(gen_ids)
            response = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            responses_list.append(response)
            print("#"*40+response+"#"*40)
        # Raw reward → shaping → normalization
        shaped_rewards = []
        original_rewards = []
        for vnf, a in zip(vnf_list, responses_list):
            raw = get_reward_from_llm_response(a, form, vnf, model_name, 1, v1, namespace)
            original_rewards.append(raw)
            shaped = shape_reward(raw)
            shaped_rewards.append(shaped)

        reward_tensor = torch.tensor(shaped_rewards, dtype=torch.float32)
        reward_norm = shape_reward(reward_tensor)

        trainer.step(
            queries=list(inputs["input_ids"]),
            responses=list(response_ids_list),
            rewards=[r for r in reward_norm]
        )
        writer.add_scalar("custom_reward/raw", torch.tensor(original_rewards).mean().item(), step)
        if step % 10 == 0:
            print(f"[PPO step={step}] mean shaped reward = {reward_tensor.mean().item():.4f}")

def train_grpo(
    model_path_or_id: str,
    mop_data: List[Dict[str, str]],
    v1,  # Kubernetes client
    form: str = "Python",
    output_dir: str = "./tmp/grpo-out",
    learning_rate: float = 1e-6,
    beta_kl: float = 0.1,              # PPO의 kl_coef/target_kl 대응 (GRPOConfig.beta)
    max_new_tokens: int = 500,         # GRPOConfig.max_completion_length 대응
    temperature: float = 0.2,
    top_p: float = 0.8,
    device: Optional[str] = None,      # (참고) GRPOTrainer가 보통 accelerate로 device를 잡습니다.
    steps: int = 20,
    batch_size: int = 3,               # PPOConfig.batch_size 대응 (per_device_train_batch_size)
    grad_accum: int = 4,               # PPO의 gradient_accumulation_steps 대응
    num_generations: int = 2,          # ⭐ GRPO 핵심: 프롬프트당 completion 개수(G). 최소 2 권장
    scale_rewards: str = "group",      # "group"(기본) / "batch" / False 등
    log_completions: bool = True,
    num_completions_to_print: int = 6,
    ):
    gc.collect()
    torch.cuda.empty_cache()

    # ✅ 1) 모델/토크나이저 로드 (GRPO는 ValueHead 불필요)
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path_or_id,
        cache_dir=hf_cache_path,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16, # bnb_config에 맞춰서
    )
    model = get_peft_model(model, lora_config, autocast_adapter_dtype=False)
    model.print_trainable_parameters()
    for name, p in model.named_parameters():
        if "lora_" in name and p.dtype == torch.float32:
            p.data = p.data.to(torch.bfloat16)
    if hasattr(model, "lm_head") and model.lm_head is not None:
        model.lm_head = model.lm_head.to(torch.bfloat16)
    
    tokenizer = AutoTokenizer.from_pretrained(model_path_or_id, cache_dir=hf_cache_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # generation 안정성에 도움이 되는 경우가 많음

    # ✅ 2) GRPO는 train_dataset에 "prompt" 컬럼이 필요
    #    그리고 reward 함수에 넘길 vnf 컬럼도 같이 둡니다. :contentReference[oaicite:1]{index=1}
    #    (문서에는 "추가 컬럼은 무시"라고도 나오지만, reward 함수에는 kwargs로 전달됩니다.)
    train_rows = [{"prompt": x["mop"], "vnf": x["vnf"]} for x in mop_data]
    train_dataset = Dataset.from_list(train_rows)

    model_name = model_path_or_id.split("/")[-1] + "_GRPO"

    # ✅ 3) custom reward 함수: GRPOTrainer가 매 스텝 내부에서 호출
    #    시그니처: reward_func(prompts, completions, completions_ids, +dataset columns as kwargs, trainer_state) :contentReference[oaicite:2]{index=2}
    def reward_func(prompts, completions, vnf, trainer_state=None, **kwargs):
        """
        - prompts: List[str] (혹은 chat format이면 List[dict]…)
        - completions: List[str]  (GRPOTrainer가 생성한 텍스트)
        - vnf: List[str] (dataset column)
        return: List[float] (completion 단위 reward)
        """
        raw_rewards = []
        shaped = []

        for v, comp in zip(vnf, completions):
            raw = get_reward_from_llm_response(comp, form, v, model_name, 1, v1, namespace)
            raw_rewards.append(float(raw))
            shaped.append(float(shape_reward(raw)))

        shaped_t = torch.tensor(shaped, dtype=torch.float32)
        norm_t = shape_reward(shaped_t) 
        return [float(x) for x in norm_t.detach().cpu().tolist()]

    # ✅ 4) GRPOConfig: PPOConfig와 매핑되는 값들을 지정
    #    - beta: KL 패널티 계수(원하면 0도 가능) :contentReference[oaicite:3]{index=3}
    #    - num_generations: 프롬프트당 completion 수(G) :contentReference[oaicite:4]{index=4}
    #    - max_completion_length: max_new_tokens 대응 :contentReference[oaicite:5]{index=5}
    args = GRPOConfig(
        output_dir=output_dir,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,

        max_steps=steps,                 # PPO의 for-step 루프를 그대로 “스텝 수”로 대응
        logging_steps=1,
        report_to=["tensorboard"],
        logging_dir=os.path.join(output_dir, "logs"),

        # generation 관련
        num_generations=num_generations,
        max_completion_length=max_new_tokens,
        temperature=temperature,
        top_p=top_p,

        # KL 제어
        beta=beta_kl,

        # reward scaling (group/batch/False)
        scale_rewards=scale_rewards,

        # Because type is bf16 in bnb_config
        bf16=True,
        fp16=False,

        # completions 로그 (프린트/로깅)
        log_completions=log_completions,
        num_completions_to_print=num_completions_to_print,
    )

    # ✅ 5) Trainer 구성 & 학습
    
    trainer = GRPOTrainer(
        model=model,
        args=args,
        reward_funcs=reward_func,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )
    with torch.autocast("cuda"):
        trainer.train()

    # 모델 저장 (LoRA adapter 포함)
    trainer.save_model(output_dir)
    print(f"\n✅ GRPO training finished. Saved to: {output_dir}")

###############################################
# 2. ONLINE DPO TRAINING LOOP
###############################################
def train_dpo(
    model_path_or_id: str,
    dpo_rows: List[Dict[str, str]],
    output_dir: str = "./tmp/dpo-out",
    per_device_train_batch_size: int = 2,
    learning_rate: float = 2e-6,
    num_train_epochs: int = 1,
    beta: float = 0.1,
    device: Optional[str] = None,
    logging_steps: int = 1,
):
    """
    dpo_rows(list of dict)로 Dataset 만든 뒤 TRL의 DPOTrainer로 학습합니다.
    """
    gc.collect()
    torch.cuda.empty_cache()
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) Dataset 구성
    train_dataset = Dataset.from_list(dpo_rows)

    # 2) 모델/토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(model_path_or_id, cache_dir=hf_cache_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = AutoModelForCausalLM.from_pretrained(model_path_or_id, quantization_config=bnb_config,device_map="auto",cache_dir=hf_cache_path)
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 3) DPOConfig 설정
    # - beta: reference 대비 정책 변화(KL 성격)를 조절하는 중요한 하이퍼파라미터
    # - remove_unused_columns=False: prompt/chosen/rejected 컬럼을 trainer가 그대로 사용하도록 유지
    dpo_config = DPOConfig(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        beta=beta,
        remove_unused_columns=False,
        logging_steps=logging_steps,
        logging_dir=output_dir+'/logs',
        report_to='tensorboard',
        save_steps=200,
    )

    # 4) Trainer 생성 및 학습
    trainer = DPOTrainer(
        model=model,
        args=dpo_config,
        processing_class=tokenizer,   # tokenizer를 넘겨주면 내부에서 토큰화/패딩 처리
        train_dataset=train_dataset,
    )

    trainer.train()
    trainer.save_model(output_dir)
    return output_dir

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--Ansible', action='store_true')
    argparser.add_argument('--Python', action='store_true')
    #argparser.add_argument('--RAG', action='store_true')
    #argparser.add_argument('--method', default = 'dpo', choices = ['dpo', 'ppo'])
    argparser.add_argument('--load-data', type=str, default=None, help='Path to the DPO dataset to load')
    argparser.add_argument('--test', action='store_true', help='Test with small number of MOPs for quick run')

    argparser=argparser.parse_args()
    mop_file_path = '../mop/Intergrated/'
    system_name='Kubernetes'
    config.load_kube_config()
    v1 = client.CoreV1Api()
    apps_v1 = client.AppsV1Api()
    form='Python' if argparser.Python else 'Ansible'

    mop_data = read_mop_file(mop_file_path, test=argparser.test)
    if argparser.load_data is not None:
        import pickle
        with open(argparser.load_data, 'rb') as f:
            input_io_pairs = pickle.load(f)
    else:
        input_io_pairs = generate_io_pairs(
            model_path_or_id='Qwen/Qwen2.5-7B-Instruct',
            mop_data=mop_data,
            v1=v1,
            form=form,
            max_new_tokens=1000,
        )
        with open('tmp/dpo_dataset.pkl' , 'wb') as f:
            import pickle
            pickle.dump(input_io_pairs, f)
    '''evaluate_llm(
        model_path_or_id='Qwen/Qwen2.5-7B-Instruct',
        mop_data=mop_data,
        v1=v1,
        form=form,
        max_new_tokens=1000,
    )'''
    train_dpo(
        model_path_or_id='Qwen/Qwen2.5-7B-Instruct',
        dpo_rows=input_io_pairs,
        output_dir='./tmp/dpo_qwen2.5_7b_k8s',
        per_device_train_batch_size=1,
        learning_rate=2e-6,
        num_train_epochs=3,
        beta=0.1,
    )
    '''train_ppo(
        model_path_or_id='./tmp/dpo_qwen2.5_7b_k8s',
        mop_data=mop_data,
        v1=v1,
        form=form,
        output_dir='./tmp/ppo_dpo_qwen2.5_7b_k8s',
    )'''
    train_grpo(
        model_path_or_id='./tmp/dpo_qwen2.5_7b_k8s',
        mop_data=mop_data,
        v1=v1,
        form=form,
        output_dir='./tmp/grpo_dpo_qwen2.5_7b_k8s',
        steps=20,
        batch_size=3,
        grad_accum=2,
        num_generations=3,
    )
    evaluate_llm(
        model_path_or_id='./tmp/grpo_dpo_qwen2.5_7b_k8s',
        mop_data=mop_data,
        v1=v1,
        form=form,
        max_new_tokens=1000,
    )
 