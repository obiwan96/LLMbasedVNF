from langchain_community.llms import Ollama
#from langchain.llms import OpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from secret import OPENAI_API_KEY, JUMP_HOST_IP, JUMP_HOST_PWD
from already_done import already_done
#from langchain.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI # to use o3-mini, use langchain_openai 0.3.x 
import argparse
from kubernetes import client, config, stream

from docx import Document
import os
os.environ["TRANSFORMERS_CACHE"]        = "/storage1/hf_cache/transformers"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"          # 선택
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import openstack
import pytz
import time
from tqdm import tqdm
from datetime import datetime
import logging
import time

#from RAG import RAG
from prompt import prompt, namespace
from openstack_config import * 
from kubernetes_config import *
import sys
import json

from transformers import AutoTokenizer
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
import torch

logging.getLogger("paramiko").setLevel(logging.CRITICAL) 


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--OpenStack', action='store_true')
    argparser.add_argument('--K8S', action='store_true')
    argparser.add_argument('--Ansible', action='store_true')
    argparser.add_argument('--Python', action='store_true')
    argparser.add_argument('--RAG', action='store_true')
    argparser.add_argument('--skip', action='store_true', help= "Skip MOPs in 'already_don.py'. Using when terminating unexpected")
    argparser.add_argument('--test', action='store_true', help='Test only 3 MOPs for testing')
    argparser.add_argument('--no-log', action='store_true', help= 'Do not log the result')
    argparser.add_argument('--judge', action='store_true', help= 'judge LLM in RAG')

    argparser=argparser.parse_args()
    mop_file_path = '../mop/Intergrated/'
    system_name='OpenStack' if argparser.OpenStack else 'Kubernetes'
    form='Python' if argparser.Python else 'Ansible'
    
    # TRL-PPO for RLHF
    # TRL 0.10.1 only support gpt-2
    model = 'EleutherAI/gpt-neox-20b' 
    tokenizer = AutoTokenizer.from_pretrained(model)
    max_len = tokenizer.model_max_length
    tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})
    llm_model = AutoModelForCausalLMWithValueHead.from_pretrained(model)
    model_ref = AutoModelForCausalLMWithValueHead.from_pretrained(model)
    # PPO Configuration
    ppo_config = PPOConfig(
        model_name=model,
        batch_size=1,
        forward_batch_size=1,
        log_with=None  # or "wandb"
    )
    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=llm_model,
        ref_model=model_ref,
        tokenizer=tokenizer
    )

    # These are same as main 
    prompts_1, prompts_2, example_code = prompt(form, system_name) # Prompots for each language and system.
    #mop_list = [file_name for file_name in os.listdir(mop_file_path) if system_name in file_name ]
    mop_list = [file_name for file_name in os.listdir(mop_file_path)]
    if argparser.test:
        mop_list=mop_list[:3]
    logging_ = not argparser.no_log
    #openai_client = OpenAI(api_key=OPENAI_API_KEY)

    if argparser.OpenStack:
        cloud_name = 'postech_cloud'
        conn = openstack.connect(cloud=cloud_name)
    elif argparser.K8S:
        config.load_kube_config()
        v1 = client.CoreV1Api()
        apps_v1 = client.AppsV1Api()
    
    logging_file = 'RL_log.txt'
    logging_file_for_vnf = 'RL_log_vnf.txt'
    with open(logging_file_for_vnf, 'w') as f:
        f.write('')
   
    if argparser.RAG:
        db_list=['RAG/stackoverflow_docs.db']
        if argparser.OpenStack:
            db_list.append('RAG/openstack_docs.db')
            if argparser.python:
                db_list.append('RAG/openstacksdk_docs.db')
        if argparser.K8S:
            db_list.append('RAG/kubernetes_docs.db')
        if argparser.Ansible:
            db_list.append('RAG/ansible_docs.db')
        collection, embed_model = RAG.RAG_init(db_list)
        if argparser.judge:
            judge_llm_model_name='llama3.3'
            judge_LLM = Ollama(model=judge_llm_model_name, num_ctx=num_ctx_list[judge_llm_model_name])
    all_mop_num = len(mop_list)
    first_create_success_num = {}
    first_config_success_num = {}
    create_success_num = {'ko':{}, 'en':{}}
    config_success_num = {'ko':{}, 'en':{}}
    success_num_by_vnf={}
    process_time = {}
    vm_num = {}
    total_start_time = time.time() 
    target_datetime = datetime.now(pytz.utc)
    for mop_file in tqdm(mop_list):
        if argparser.skip:    
            if mop_file in already_done:
                continue
        mop_suc_num=0
        vnf = mop_file.split('_')[0]
        lang = mop_file.split('_')[3]
        action = mop_file.split('_')[2] # confiugration action.
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
        if vnf not in vm_num:
            vm_num[vnf] = 1
        else:
            vm_num[vnf] += 1
        if vnf not in success_num_by_vnf:
            success_num_by_vnf[vnf] = {'total_num' : 1, 'success_num': 0}
        else:
            success_num_by_vnf[vnf]['total_num'] += 1
        
        # Parameters for VM creation
        vm_name = 'vm-'+vnf+'-'+str(vm_num[vnf])
        mop=''
        assert (mop_file.endswith('.docx'))
        doc = Document(mop_file_path + mop_file)
        for para in doc.paragraphs:
            mop += para.text + '\n'
        if logging_:
            with open(logging_file, 'a') as f:
                f.write(f" --- Model: {model}, MOP: {mop_file}, VNF: {vnf}, VM: {vm_num[vnf]}\n")
        spend_time = [0, 0, 0] # [llm response time, vm creation time, vm configuration time]
        start_time = time.time()

        input_ids = tokenizer(prompts_1+prompts_2+ mop +additional_action+ example_code,max_length= max_len,  return_tensors = "pt").input_ids
        query_tensors = [input_ids[i] for i in range(input_ids.size(0))]
        response_ids = ppo_trainer.generate(query_tensors, max_new_tokens=20)
        llm_response = tokenizer.decode(response_ids[0], skip_special_tokens=True)

        already_success = False
        spend_time[0] += time.time()-start_time
        start_time = time.time()
        if logging_:
            with open(logging_file, 'a') as f:
                f.write(f" --- Trial: {_+1}\n")
                f.write(llm_response+'\n\n')
        if form=='Python':
            if system_name=='OpenStack':
                test_result, server_or_message = test_creation_python_OpenStack(llm_response, vnf, model, vm_num[vnf])
            elif system_name=='Kubernetes':
                test_result, server_or_message = test_creation_python_K8S(llm_response, vnf, model, vm_num[vnf], _, v1, namespace)
        else:
            test_result, server_or_message = test_creation_ansible_K8S(llm_response, vnf, model, vm_num[vnf], v1, 600)
        spend_time[1] = time.time()-start_time
        if test_result == 0:
            # TODO: OpenStack module still return 'True' if succeed.
            if system_name=='OpenStack':
                try:
                    server = conn.compute.wait_for_server(server_or_message)
                except:
                    error_message="The 'create_vm' function does not return 'server' object. "
                    if logging_:
                        with open(logging_file, 'a') as f:
                            f.write(error_message+'\n')
                    continue
            # VM creation success.
            # Now, test the configuration
            # print('creation success')
            create_success_num[lang][model] += 1

            start_time = time.time()

            # Test configuration here. Need change for K8S and Ansible.
            # Todo: Needs develop SFC checking module.

            if system_name=='OpenStack':
                second_test_result = test_openstack_configuration(server_or_message, vnf, model, vm_num[vnf], conn, None)
            if system_name=='Kubernetes':
                second_test_result, second_message = test_K8S_configuration(server_or_message, vnf, v1, namespace)
            spend_time[2] = time.time()-start_time
            if system_name=='OpenStack':
                conn.delete_server(server_or_message.id)
            if system_name=='Kubernetes':
                delete_pod(v1, server_or_message, namespace)
            if second_test_result == 0:
                # TODO: OpenStack module still return 'True' if succeed.
                process_time[model].append(spend_time)
                success_num_by_vnf[vnf]['success_num'] += 1
                config_success_num[lang][model] += 1
                if logging_:
                    with open(logging_file, 'a') as f:
                        f.write('Test succeed\n')
                print(f'VM creation and configuration both Succeed! model: {model}, vnf: {vnf}')
                mop_suc_num+=1
                reward = 1
                break
            else:
                # VM Config test failed.
                if logging_:
                    with open(logging_file, 'a') as f:
                        f.write('Config test failled. Results:\n')
                        f.write(errorcode_dict[second_test_result]+'\n')
                # creation success, but configuration failed. 
                # reward is -0.5
                reward = -0.5
        else:
            # VM Creation failed
            if logging_:
                with open(logging_file, 'a') as f:
                    f.write('VM Creation test failled. Results:\n')
                    f.write(errorcode_dict[test_result]+'\n')
            reward = -1
        if system_name=='OpenStack':
            # Delete all VMs created after the target time
            delete_vms_after(conn, target_datetime)
        elif system_name=='Kubernetes':
            delete_all_pods(v1, apps_v1)
        # change next print operations to write in the logging_file
        if logging_:
            with open(logging_file, 'a') as f:
                f.write('Middle report\n')
                f.write(f"First VM Create success: {first_create_success_num[model]}\n")
                f.write(f"Korean MOP - VM Create success: {create_success_num['ko'][model]}\n")
                f.write(f"English MOP - VM Create success: {create_success_num['en'][model]}\n")
                f.write(f"Total VM Create Success: {create_success_num['ko'][model]+create_success_num['en'][model]}\n")
                f.write(f"First VNF Config success: {first_config_success_num[model]}\n")
                f.write(f"Korean MOP - VNF Config success: {config_success_num['ko'][model]}\n")
                f.write(f"English MOP - VNF Config success: {config_success_num['en'][model]}\n")
                f.write(f"Total VNF Config Success: {config_success_num['ko'][model]+config_success_num['en'][model]}\n")
                if len(process_time[model]) > 0:
                    tot_num=len(process_time[model])
                    pr_ti = process_time[model]
                    f.write(f"Average LLM process time: {sum([i[0] for i in pr_ti])/tot_num} seconds\n")
                    f.write(f"Average VM creation time: {sum([i[1] for i in pr_ti])/tot_num} seconds\n")
                    f.write(f"Average VM configuration time: {sum([i[2] for i in pr_ti])/tot_num} seconds\n")
                f.write(f"--------------------------------------------\n")
        with open(logging_file_for_vnf, 'a')as f:
            f.write(f" --- MOP: {mop_file}, success num: {mop_suc_num}\n")
        
        # PPO training
        stats = ppo_trainer.step(
            query_tensors,          # ← List[Tensor(seq_len,)]
            [response_ids[0]],       # ← List[Tensor(new_seq_len,)]
            [torch.tensor(reward)]                 # ← List[float]
        )
        print(f"Reward: {reward}, Stats: {stats}")
    
    end_time = time.time()
    #conn.compute.delete_server(floating_server.id)
    print('Total report')
    print(f"Total MOPs: {all_mop_num}")
    print(f"Model: {model},")
    print(f"First VM Create success: {first_create_success_num[model]}")
    print(f"Korean MOP - VM Create success: {create_success_num['ko'][model]}")
    print(f"English MOP - VM Create success: {create_success_num['en'][model]}")
    print(f"Total VM Create Success: {create_success_num['ko'][model]+create_success_num['en'][model]}")
    print(f"First VNF Config success: {first_config_success_num[model]}")
    print(f"Korean MOP - VNF Config success: {config_success_num['ko'][model]}")
    print(f"English MOP - VNF Config success: {config_success_num['en'][model]}")
    print(f"Total VNF Config Success: {config_success_num['ko'][model]+config_success_num['en'][model]}")
    if len(process_time[model]) > 0:
        tot_num=len(process_time[model])
        pr_ti = process_time[model]
        print(f"Average LLM process time: {sum([i[0] for i in pr_ti])/tot_num} seconds")
        print(f"Average VM creation time: {sum([i[1] for i in pr_ti])/tot_num} seconds")
        print(f"Average VM configuration time: {sum([i[2] for i in pr_ti])/tot_num} seconds")
    print(f"--------------------------------------------------")
    with open(logging_file_for_vnf, 'a')as f:
        for vnf in success_num_by_vnf:
            f.write(f"VNF: {vnf}, Total MOPs: {success_num_by_vnf[vnf]['total_num']}, Success: {success_num_by_vnf[vnf]['success_num']}\n")
    
    print(f'Total execution time:{(end_time-total_start_time)/60/60} hours')