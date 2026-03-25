import torch
from transformers import BitsAndBytesConfig
import os, random
from docx import Document
from prompt import prompt
from openstack_config import * 
from kubernetes_config import *
from typing import Dict
prompts_1, prompts_2, example_data = prompt('Python', 'Kubernetes') # Prompots for each language and system.
good_example, good_example_prefix, goood_example_suffix = example_data
hf_cache_path = '/storage3/hf_cache/'

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
SYSTEM_MESSAGE = (
    "You are a helpful assistant that generates Python code for "
    "Kubernetes automation. "
    "Your output must be executable Python code only. "
    "Do not include explanations, markdown, or comments outside code. "
    "The code must create and configure Kubernetes Pods exactly as requested."
)

def build_enc(tokenizer, prompt_text: str, max_input_tokens: int = 8192):
    messages = [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": prompt_text},
    ]
    # add_generation_prompt=True 가 핵심: "이제 모델이 답할 차례"를 붙여줌
    enc = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        truncation=True,
        return_tensors="pt",
        max_length=max_input_tokens,
    )
    if hasattr(enc, "input_ids"):
        enc = enc["input_ids"]
    return {"input_ids": enc}

def read_mop_file(file_path: str, system_name: str, test:int = None) -> list[str]:
    mop_list = [file_name for file_name in os.listdir(file_path)]

    # Just be simple
    if test:
        mop_len = len(mop_list)
        mop_list = random.sample(mop_list, len(mop_list)) # Shuffle the list
        mop_list = mop_list[:int(mop_len/test)]  # Test with 1/test MOPs
        #mop_list = mop_list[:5] # Test with 5 MOPs

    mop_data = []
    for mop_file in mop_list:
        #print(mop_file)
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
        doc = Document(file_path + mop_file)
        for para in doc.paragraphs:
            mop += para.text + '\n'
        single_mop = {'vnf': vnf, 'lang': lang, 'mop': prompts_1+prompts_2+ mop +additional_action+ example_code, 'vm_name': vm_name}
        mop_data.append(single_mop)
    return mop_data        

def get_response_from_llm(model, tokenizer, max_new_tokens, temperature, top_p, prompt, do_sample=False) -> str:
    #assert tokenizer.name_or_path == model.config.name_or_path
    enc = build_enc(tokenizer, prompt, max_input_tokens=8192)
    input_ids = enc["input_ids"]
    if hasattr(input_ids, "input_ids"):
        input_ids = input_ids["input_ids"]
    first_device = next(model.parameters()).device
    input_ids = input_ids.to(first_device)
    attention_mask = torch.ones_like(input_ids)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    generation_config = copy.deepcopy(model.generation_config)
    generation_config.max_new_tokens = max_new_tokens
    generation_config.max_length = None
    generation_config.pad_token_id = tokenizer.eos_token_id
    generation_config.eos_token_id = tokenizer.eos_token_id
    generation_config.repetition_penalty = 1.1
    if do_sample:
        generation_config.temperature = temperature
        generation_config.top_p = top_p
        generation_config.top_k = 50
    with torch.no_grad():
        out_ids = model.generate(
            input_ids=input_ids.to(device),
            attention_mask=attention_mask.to(device),
            generation_config=generation_config,
            do_sample=do_sample,
        )

    input_len = input_ids.shape[1]
    gen_ids = out_ids[0][input_len:]
    answer = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    return answer

def get_reward_from_llm_response(answer: str, form: str, vnf: str, model_name: str, vm_num: dict, k8s_client, namespace: str, logging:bool=False, use_whole_code = False) -> float:
    logging_file = f'tmp/rl_llm_response_log.txt'
    answer = answer.split("Human:")[0].strip()
    second_message = None
    if logging:        
        with open(logging_file, 'a') as f:
            f.write(f" --- Trial for VNF: {vnf}, Model: {model_name}, VM number: {vm_num[vnf]} --- \n")
            f.write(answer+'\n\n')
    v1, apps_v1 = k8s_client
    if form == 'Python':
        test_result, server_or_message = test_creation_python_K8S(answer, vnf, model_name, vm_num[vnf], 1, v1, namespace, use_whole_response=use_whole_code)
    else:
        test_result, server_or_message = test_creation_ansible_K8S(answer, vnf, model_name, vm_num[vnf], v1, 600, use_whole_code=use_whole_code)
    if test_result == 0: # Creation Success
        second_test_result, second_message = test_K8S_configuration(server_or_message, vnf, v1, namespace)
        delete_pod(v1, server_or_message, namespace)
        #print(f'## Creation success, status code: {second_test_result}')
        with open(logging_file, 'a') as f:
            f.write(f'## Creation success, status code: {second_test_result}')
        if second_test_result == 0:
            # Both Creation and Configuration Success
            reward = 1.0
        # Now, make reward based on status code.
        elif second_test_result == 1: #Can't find code in LLM's response
            reward = 0.0
        elif second_test_result == 2: #VNF does not work as intended
            reward = 0.9
        
        elif second_test_result in [31, 32, 33] or is_k8s_config_error_code(second_test_result):
            if second_test_result == 31: #Can't find Pod(wrong namespace or wrong host name)
                reward = 0.6
            elif second_test_result == 32: # code need to include 'sleep infinity' to container keep on.
                reward = 0.8
            elif second_test_result == 33:
                # Container process failed after deployment.
                reward = 0.45
            elif second_test_result in [60, 61]:
                # Kubelet/API connectivity timeout: usually infrastructure-side and less attributable to generated code.
                reward = 0.65
            elif second_test_result in [62, 63, 69]:
                # API failure but not timeout (or unknown class): medium severity.
                reward = 0.55
            elif second_test_result == 64:
                # Error-like pattern found in pod logs: often indicates actual config/runtime issue.
                reward = 0.5
            else:
                # Fallback for backward compatibility (legacy 30, etc.)
                reward = 0.5
        else:
            print('## Unknown error code from configuration test:', second_test_result)
            return None
    else:
        with open(logging_file, 'a') as f:
            f.write('## Creation failed')
        if test_result == 1: #Can't find code in LLM's response
            reward = 0.0
        elif test_result == 10: #Code parsing fail
            reward = 0.05
        elif test_result == 14: #'create_pod' does not exist in code
            reward = 0.1
        elif test_result == 13: # Error while import 'create_pod'
            reward = 0.2
        elif test_result in [11, 12]: # Need change in code. include none-kubernetes cde or infinite loop
            reward = 0.3
        elif test_result in [20, 21, 22, 23]: # Error occured during running 'pod_creation'
            if test_result == 23: # Ansible run status error
                reward = 0.35
            elif test_result == 21: # Little bigger error occure than 20. 20 has error log
                reward = 0.4
            else: # 20, 22. Small error occur during pod creation
                reward = 0.45
        elif test_result in [41, 43, 51, 90, 91]:
            if test_result == 41:
                reward = 0.6
            elif test_result == 43:
                reward = 0.5
            elif test_result in [90, 91]: # Code running timeout
                reward = 0.7
            elif test_result == 51: # VNF configuration code return False
                reward = 0.75
        else:
            print('## Unknown error code from configuration test:', test_result)
            return None
        #print('## Creation failed')
        second_message = server_or_message
        second_test_result = test_result
    delete_all_pods(v1, apps_v1, namespace=namespace)
    return reward, (second_test_result, second_message)

def calculate_dynamic_params(
    data_size: int,
    training_type: str = "grpo",
    using_cot: bool = False,
) -> Dict[str, int]:
    """
    데이터 양과 GPU 메모리에 따라 학습 파라미터를 동적으로 계산합니다.
    
    Args:
        data_size: 학습 데이터 개수
        training_type: "grpo", "dpo", "grpo_trajectory" 중 하나
        using_cot: CoT 방식 사용 여부 (메모리 증가)
    
    Returns:
        {"batch_size": int, "num_generations": int, "num_prompts_per_step": int, 
         "grad_accum": int, "max_new_tokens": int, "steps": int}
    """
    params = {}
    if training_type == "dpo":
        # DPO는 chosen/rejected와 reference log-prob까지 함께 계산하므로
        # GRPO보다도 시퀀스 길이와 배치 크기에 민감합니다.
        if data_size <= 100:
            params = {
                "batch_size": 1,
                "grad_accum": 8,
                "steps": 10,
            }
        elif data_size <= 500:
            params = {
                "batch_size": 1,
                "grad_accum": 16,
                "steps": 20,
            }
        else:
            params = {
                "batch_size": 1,
                "grad_accum": 16,
                "steps": 30,
            }
            
    elif training_type == "grpo" or training_type == "grpo_trajectory":
        # GRPO는 generation KV cache가 지배적이라, Unsloth를 써도
        # rollout 병렬도와 출력 길이는 보수적으로 잡아야 OOM을 피할 수 있습니다.
        
        if data_size <= 50:
            params = {
                "batch_size": 2,
                "num_generations": 4,
                "num_prompts_per_step": 2,
                "grad_accum": 4,
                "steps": 50,
            }
        elif data_size <= 200:
            params = {
                "batch_size": 1,
                "num_generations": 4,
                "num_prompts_per_step": 1,
                "grad_accum": 8,
                "steps": 80,
            }
        else:  # data_size > 200
            params = {
                "batch_size": 1,
                "num_generations": 2,
                "num_prompts_per_step": 1,
                "grad_accum": 8,
                "steps": int(data_size * 0.8), # 데이터에 비례하여 스텝 수 증가
            }
    params['max_prompt_length'] = 6500
    base_tokens = 1024+500 if using_cot else 1024
    params['max_new_tokens'] = base_tokens
    print(f"\n[UNSLOTH PARAMS] Data size: {data_size}, Type: {training_type}, CoT: {using_cot}")
    print(f"[UNSLOTH PARAMS] Calculated params: {params}")
    return params