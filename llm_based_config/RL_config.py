import torch
from transformers import BitsAndBytesConfig
import os, random
from docx import Document
from prompt import prompt
from openstack_config import * 
from kubernetes_config import *
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
    first_device = next(model.parameters()).device
    enc = {k: v.to(first_device) for k, v in enc.items()}
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
    #print("\n##>> input_len:", input_len, "output_len:", output_len, "new_tokens:", output_len - input_len)
    gen_ids = out_ids[0][input_len:]
    # 디코딩
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
        if test_result == 10: #Code parsing fail
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