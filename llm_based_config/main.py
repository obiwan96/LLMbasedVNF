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
os.environ["OLLAMA_USE_GPU"] = "true"
import openstack
import pytz
import time
from tqdm import tqdm
from datetime import datetime
import logging
import time

from RAG import RAG
from prompt import prompt, namespace
from openstack_config import * 
from kubernetes_config import *
import sys

logging.getLogger("paramiko").setLevel(logging.CRITICAL) 

def multi_agent_debate(mop_file_path, mop_list, model_list,num_ctx_list, form,system_name, prompts, logging_, maximum_tiral, ragging):
    logging_file = 'log_debate.txt'
    if system_name=='OpenStack':
        cloud_name = 'postech_cloud'
        conn = openstack.connect(cloud=cloud_name)
    elif system_name=='Kubernetes':
        config.load_kube_config()
        v1 = client.CoreV1Api()
        apps_v1 = client.AppsV1Api()

    prompts_1, prompts_2, example_code = prompts
    if ragging:
        db_list=['RAG/stackoverflow_docs.db']
        if system_name=='OpenStack':
            db_list.append('RAG/openstack_docs.db')
            if form=='Python':
                db_list.append('RAG/openstacksdk_docs.db')
        if system_name=='Kubernetes':
            db_list.append('RAG/kubernetes_docs.db')
        if form=='Ansible':
            db_list.append('RAG/ansible_docs.db')
        collection, embed_model = RAG.RAG_init(db_list)
    first_create_success_num = {}
    first_config_success_num = {}
    create_success_num = {'ko':{}, 'en':{}}
    config_success_num = {'ko':{}, 'en':{}}
    success_num_by_vnf={}
    process_time = {}
    for model in model_list:
        first_create_success_num[model] = 0
        first_config_success_num[model] = 0
        create_success_num['ko'][model] = 0
        create_success_num['en'][model] = 0
        config_success_num['ko'][model] = 0
        config_success_num['en'][model] = 0
        process_time[model] = []
    vm_num = {}
    total_start_time = time.time() 
    target_datetime = datetime.now(pytz.utc)
    for mop_file in tqdm(mop_list):
        if not (mop_file in already_done):
            continue
        llm_responses =[]
        spend_time = [0, 0, 0] # [llm response time, vm creation time, vm configuration time]
        start_time=time.time()
        mop_suc_num=0
        vnf = mop_file.split('_')[0]
        lang = mop_file.split('_')[3]
        action = mop_file.split('_')[2] # confiugration action.
        if action == 'port':
            additional_action='Block the port except 22 and 80.\n'
        elif action in ['subnet', 'block'] :
            additional_action='Block the traffic except subnets with'
            if system_name == 'OpenStack':
                additional_action+=' 10.10.10.0/24 and 10.10.20.0/24.\n'
            else:
                additional_action+=' 10.244.0.0/16.\n'
            if vnf=='nDPI':
                additional_action+= "Please use nDPI to block, following the MOP. Don't use the firewall.\n"
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
        main_model = model_list[0]
        main_llm = Ollama(model=main_model, num_ctx=num_ctx_list[model]*3)
        chat = ConversationChain(llm=main_llm, memory = ConversationBufferMemory())
        llm_response=chat.invoke(prompts_1+additional_action+prompts_2+ mop + example_code)['response']
        #llm_responses.append(llm_response) # not inlcude main llm's response, right?

        for model in model_list[1:]:
            if logging_:
                with open(logging_file, 'a') as f:
                    f.write(f" --- Model: {model}, MOP: {mop_file}, VNF: {vnf}, VM: {vm_num[vnf]}\n")
            if model.startswith('gpt'):
                llm = ChatOpenAI(temperature=0, model_name=model)
            elif model.startswith('o3'):
                llm = ChatOpenAI(model_name=model)
            else:   
                llm = Ollama(model=model, num_ctx=num_ctx_list[model])
            llm_response=llm.invoke(prompts_1+additional_action+prompts_2+ mop + example_code)
            llm_responses.append(llm_response)
        llm_response = chat.invoke('Here are results from other LLMs. Update your response based on the other answers, but do not change the original requirements I specified at the start.'+ \
                                   '\n'.join(llm_responses))['response']
        
            #print(llm_response)
        already_success = False
        for _ in range(maximum_tiral):
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
            if test_result == True:
                if system_name=='OpenStack':
                    try:
                        server = conn.compute.wait_for_server(server_or_message)
                    except:
                        error_message="The 'create_vm' function does not return 'server' object. "
                        if logging_:
                            with open(logging_file, 'a') as f:
                                f.write(error_message+'\n')
                        if _ < maximum_tiral-1:
                            start_time = time.time()
                            llm_response=chat.invoke(error_message+'Please fix it.\n')['response']
                        continue
                # VM creation success.
                # Now, test the configuration
                print('creation success')
                if not already_success:
                    create_success_num[lang][model] += 1
                    already_success = True
                #if _>0:
                #    print(f"succeed after {_+1} times trial")
                if _ == 0:
                    first_create_success_num[model] += 1

                start_time = time.time()

                # Test configuration here. Need change for K8S and Ansible.
                # Todo: Needs develop SFC checking module.

                if system_name=='OpenStack':
                    second_test_result = test_openstack_configuration(server_or_message, vnf, model, vm_num[vnf], conn, None)
                if system_name=='Kubernetes':
                    second_test_result = test_K8S_configuration(server_or_message, vnf, v1, namespace)
                spend_time[2] = time.time()-start_time
                if system_name=='OpenStack':
                    conn.delete_server(server_or_message.id)
                if system_name=='Kubernetes':
                    delete_pod(v1, server_or_message, namespace)
                if second_test_result == True:
                    process_time[model].append(spend_time)
                    success_num_by_vnf[vnf]['success_num'] += 1
                    config_success_num[lang][model] += 1
                    if _ == 0:
                        first_config_success_num[model] += 1
                    if logging_:
                        with open(logging_file, 'a') as f:
                            f.write('Test succeed\n')
                    print(f'VM creation and configuration both Succeed! model: {model}, vnf: {vnf}')
                    mop_suc_num+=1
                    break
                else:
                    # VM Config test failed.

                    # Todo: add RAG here.
                    if logging_:
                        with open(logging_file, 'a') as f:
                            f.write('Config test failled. Results:\n')
                            f.write(str(second_test_result)+'\n')
                    if _ < maximum_tiral-1:
                        start_time = time.time()
                        if argparser.RAG and not 'It should return' in str(second_test_result):
                            # VNF configuration related docs are not crawled yet. only StackOverflow
                            if 'Error occurs' in str(second_test_result):
                                error_logs = '\n'.join(return_error_logs(str(second_test_result)))
                                retrieved_texts=RAG.RAG_search(error_logs, collection, embed_model)
                                if logging_:
                                    with open(logging_file_rag, 'a') as f:
                                        f.write('------------------------------')
                                        f.write('RAG input:\n')
                                        f.write(error_logs+'\n')
                                        f.write('#####')
                                        f.write('RAG results:\n')
                                        f.write(retrieved_texts+'\n')
                            else:
                                retrieved_texts=RAG.RAG_search(second_test_result, collection, embed_model)
                                if logging_:
                                    with open(logging_file_rag, 'a') as f:
                                        f.write('------------------------------')
                                        f.write('RAG input:\n')
                                        f.write(str(second_test_result)+'\n')
                                        f.write('#####')
                                        f.write('RAG results:\n')
                                        f.write(retrieved_texts+'\n')
                            second_test_result= str(second_test_result) + '\n And here is a related document. Please refer to it.' + retrieved_texts
                            if logging_:
                                with open(logging_file, 'a') as f:
                                    f.write ('RAG results:\n')
                                    f.write(retrieved_texts+'\n')
                        if system_name=='OpenStack':
                            llm_response=chat.invoke('When I run your code, I can successfully create VM, '+ \
                                'but VNF configuration is failed. I got this error message. Please correct the code and return the updated version.\n'+str(second_test_result))['response']
                        if system_name == 'Kubernetes':
                            llm_response=chat.invoke('When I run your code, I can successfully create Pod, '+ \
                                'but VNF is not installed correctly as intended. Please correct the code and return the updated version by refering MOP again.\n'+str(second_test_result))['response']
            else:
                # VM Creation failed
                if logging_:
                    with open(logging_file, 'a') as f:
                        f.write('VM Creation test failled. Results:\n')
                        f.write(str(server_or_message)+'\n')
                if _ < maximum_tiral-1:
                    #print(test_result)
                    #print(f'{_+2} try')
                    start_time = time.time()
                    if argparser.RAG and ('Ansible runner status:' in str(server_or_message) or 'but Pod got error: ' in str(server_or_message)):

                        # ToDo: currently consider only K8S. need to fix to consider OpenStack also.
                        error_start = check_log_error(str(server_or_message))
                        if error_start:
                            error_logs = '\n'.join(return_error_logs(str(server_or_message)))
                            retrieved_texts=RAG.RAG_search(error_logs, collection, embed_model)
                            if logging_:
                                with open(logging_file_rag, 'a') as f:
                                    f.write('------------------------------')
                                    f.write('RAG input:\n')
                                    f.write(error_logs+'\n')
                                    f.write('#####')
                                    f.write('RAG results:\n')
                                    f.write(retrieved_texts+'\n')
                        else:
                            retrieved_texts=RAG.RAG_search(server_or_message, collection, embed_model)
                            if logging_:
                                with open(logging_file_rag, 'a') as f:
                                    f.write('------------------------------')
                                    f.write('RAG input:\n')
                                    f.write(str(server_or_message)+'\n')
                                    f.write('#####')
                                    f.write('RAG results:\n')
                                    f.write(retrieved_texts+'\n')

                        server_or_message= str(server_or_message) + '\n And here is a related document. Please refer to it.' + retrieved_texts
                        if logging_:
                            with open(logging_file, 'a') as f:
                                f.write ('RAG results:\n')
                                f.write(retrieved_texts+'\n')
                    if form=='Python' and 'has no attribute ' in str(server_or_message):
                        func_name = str(server_or_message).split("'")[3]
                        #print(func_name)
                        llm_response=chat.invoke(f"Your code doesn't have '{func_name}' function. Please put the code inside it. Also, it should take 'pod_name', 'namespace', and 'image_name' as input and return True if configuration succeed. Don't use the other function name.")['response']
                    elif form=='Python' and 'positional arguments but 3 were given' in str(server_or_message):
                        llm_response=chat.invoke(f"'create_pod' function should take 'pod_name', 'namespace', and 'image_name' as input and return True if configuration succeed.")['response']
                    else:
                        llm_response=chat.invoke('When I run your code, I got this error message, and failed to create VM. Please correct the code and return the updated version.\n'+str(server_or_message))['response']
            if system_name=='OpenStack':
                # Delete all VMs created after the target time
                delete_vms_after(conn, target_datetime)
            elif system_name=='Kubernetes':
                delete_all_pods(v1, apps_v1)
        # change next print operations to write in the logging_file
        if logging_:
            with open(logging_file, 'a') as f:
                f.write('Middle report\n')
                for model in model_list:
                    f.write(f"Model: {model},\n")
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
                    f.write(f"-------------------------------------------------------------\n")
    end_time = time.time()

    print(f'Total execution time:{(end_time-total_start_time)/60/60} hours')

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--OpenStack', action='store_true')
    argparser.add_argument('--K8S', action='store_true')
    argparser.add_argument('--Ansible', action='store_true')
    argparser.add_argument('--Python', action='store_true')
    argparser.add_argument('--gpt', action='store_true')
    argparser.add_argument('--llama', action='store_true')
    argparser.add_argument('--code-llm', action='store_true')
    argparser.add_argument('--RAG', action='store_true')
    argparser.add_argument('--skip', action='store_true', help= "Skip MOPs in 'already_don.py'. Using when terminating unexpected")
    argparser.add_argument('--test', action='store_true', help='Test only 3 MOPs for testing')
    argparser.add_argument('--repre-llms', action='store_true', help= 'Using only representative LLMs (except GPT)')
    argparser.add_argument('--no-log', action='store_true', help= 'Do not log the result')
    argparser.add_argument('--max-limit', help= 'Max limit of MOPs to test', type=int, default=3)
    argparser.add_argument('--debate', action='store_true', help= 'multi-agent-debate')

    argparser=argparser.parse_args()
    if argparser.OpenStack:
        mop_file_path = '../mop/OpenStack_v3/'
    elif argparser.K8S:
        mop_file_path = '../mop/K8S_v1/'
    # Let's use intergrated MOPs!
    mop_file_path = '../mop/Intergrated/'
    system_name='OpenStack' if argparser.OpenStack else 'Kubernetes'
    form='Python' if argparser.Python else 'Ansible'
    prompts_1, prompts_2, example_code = prompt(form, system_name) # Prompots for each language and system.
    #mop_list = [file_name for file_name in os.listdir(mop_file_path) if system_name in file_name ]
    mop_list = [file_name for file_name in os.listdir(mop_file_path)]
    if argparser.test:
        mop_list=mop_list[:3]
    os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
    logging_ = not argparser.no_log
    #openai_client = OpenAI(api_key=OPENAI_API_KEY)

    maximum_tiral = argparser.max_limit # 3 is default. give 2 more chance
    if argparser.OpenStack:
        cloud_name = 'postech_cloud'
        conn = openstack.connect(cloud=cloud_name)
    elif argparser.K8S:
        config.load_kube_config()
        v1 = client.CoreV1Api()
        apps_v1 = client.AppsV1Api()
    
    logging_file = 'log.txt'
    logging_file_rag = 'log_rag.txt'
    logging_file_for_vnf = 'log_vnf.txt'
    with open(logging_file_for_vnf, 'w') as f:
        f.write('')
    model_list=[]
    if argparser.gpt:
        model_list.extend(["gpt-3.5-turbo", "gpt-4o", "o3-mini"])
        model_list = ["o3-mini"]
    if argparser.llama:
        model_list.extend(["llama3.3", "codellama:70b"])
    if argparser.code_llm:
        model_list.extend(["o3-mini", "codellama:70b", "qwen2.5-coder:32b", "codegemma:7b"])
    if argparser.repre_llms:
        model_list = ["llama3.3", "qwen2.5-coder:32b", "gemma3:27b", "mistral", "phi4"]
    elif not argparser.gpt and not argparser.llama and not argparser.code_llm:
        model_list= [# "gpt-4o", "o3-mini",  #The money ran out too fast, so left it out for a while
                     "llama3.3", "qwen2.5-coder:32b", "deepseek-r1:70b",
                      "gemma3:27b", "qwq", "phi4", "mistral"]
    num_ctx_list = {
        "llama3.3" : 10000, #131072
        "llama3.1:70b" : 8192,
        "qwen2" :8192,
        "qwen2:72b" : 32768,
        "gemma2" : 8192,
        "gemma2:27b" : 8192,
        "gemma3:27b" : 10000, #131072
        "deepseek-r1:70b" : 10000, #131072
        "qwen2.5:72b" : 32768,
        "codellama:70b" : 2048,
        "qwen2.5-coder:32b" : 10000, #32768
        "codegemma:7b" : 8192,
        "phi4":10000, #16384
        "mistral":10000, #32768
        "qwq": 10000 #40960
    }
    if argparser.debate:
        model_list = ["llama3.3", "qwen2.5-coder:32b", "gemma3:27b", "qwq"]
        prompts = (prompts_1, prompts_2, example_code)
        multi_agent_debate(mop_file_path, mop_list, model_list,num_ctx_list, form,system_name, prompts, logging_, maximum_tiral, argparser.RAG)
        sys.exit()

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
    all_mop_num = len(mop_list)
    first_create_success_num = {}
    first_config_success_num = {}
    create_success_num = {'ko':{}, 'en':{}}
    config_success_num = {'ko':{}, 'en':{}}
    success_num_by_vnf={}
    process_time = {}
    for model in model_list:
        first_create_success_num[model] = 0
        first_config_success_num[model] = 0
        create_success_num['ko'][model] = 0
        create_success_num['en'][model] = 0
        config_success_num['ko'][model] = 0
        config_success_num['en'][model] = 0
        process_time[model] = []
    vm_num = {}
    total_start_time = time.time() 
    target_datetime = datetime.now(pytz.utc)
    #floating_server = make_new_floating_ip(conn)
    #if not floating_server:
    #    print('Make flaoting IP failed')
    #    exit()
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
        for model in model_list:
            if logging_:
                with open(logging_file, 'a') as f:
                    f.write(f" --- Model: {model}, MOP: {mop_file}, VNF: {vnf}, VM: {vm_num[vnf]}\n")
            spend_time = [0, 0, 0] # [llm response time, vm creation time, vm configuration time]
            start_time = time.time()
            if model.startswith('gpt'):
                llm = ChatOpenAI(temperature=0, model_name=model)
            elif model.startswith('o3'):
                llm = ChatOpenAI(model_name=model)
            else:   
                llm = Ollama(model=model, num_ctx=num_ctx_list[model])
            chat = ConversationChain(llm=llm, memory = ConversationBufferMemory())
            #llm_response=llm.invoke(prompts+mop)
            llm_response=chat.invoke(prompts_1+prompts_2+ mop +additional_action+ example_code)['response']
            #print(llm_response)
            already_success = False
            for _ in range(maximum_tiral):
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
                if test_result == True:
                    if system_name=='OpenStack':
                        try:
                            server = conn.compute.wait_for_server(server_or_message)
                        except:
                            error_message="The 'create_vm' function does not return 'server' object. "
                            if logging_:
                                with open(logging_file, 'a') as f:
                                    f.write(error_message+'\n')
                            if _ < maximum_tiral-1:
                                start_time = time.time()
                                llm_response=chat.invoke(error_message+'Please fix it.\n')['response']
                            continue
                    # VM creation success.
                    # Now, test the configuration
                    print('creation success')
                    if not already_success:
                        create_success_num[lang][model] += 1
                        already_success = True
                    #if _>0:
                    #    print(f"succeed after {_+1} times trial")
                    if _ == 0:
                        first_create_success_num[model] += 1

                    start_time = time.time()

                    # Test configuration here. Need change for K8S and Ansible.
                    # Todo: Needs develop SFC checking module.

                    if system_name=='OpenStack':
                        second_test_result = test_openstack_configuration(server_or_message, vnf, model, vm_num[vnf], conn, None)
                    if system_name=='Kubernetes':
                        second_test_result = test_K8S_configuration(server_or_message, vnf, v1, namespace)
                    spend_time[2] = time.time()-start_time
                    if system_name=='OpenStack':
                        conn.delete_server(server_or_message.id)
                    if system_name=='Kubernetes':
                        delete_pod(v1, server_or_message, namespace)
                    if second_test_result == True:
                        process_time[model].append(spend_time)
                        success_num_by_vnf[vnf]['success_num'] += 1
                        config_success_num[lang][model] += 1
                        if _ == 0:
                            first_config_success_num[model] += 1
                        if logging_:
                            with open(logging_file, 'a') as f:
                                f.write('Test succeed\n')
                        print(f'VM creation and configuration both Succeed! model: {model}, vnf: {vnf}')
                        mop_suc_num+=1
                        break
                    else:
                        # VM Config test failed.

                        # Todo: add RAG here.
                        if logging_:
                            with open(logging_file, 'a') as f:
                                f.write('Config test failled. Results:\n')
                                f.write(str(second_test_result)+'\n')
                        if _ < maximum_tiral-1:
                            start_time = time.time()
                            if argparser.RAG and not 'It should return' in str(second_test_result):
                                # VNF configuration related docs are not crawled yet. only StackOverflow
                                if 'Error occurs' in str(second_test_result):
                                    error_logs = '\n'.join(return_error_logs(str(second_test_result)))
                                    retrieved_texts=RAG.RAG_search(error_logs, collection, embed_model)
                                    if logging_:
                                        with open(logging_file_rag, 'a') as f:
                                            f.write('------------------------------')
                                            f.write('RAG input:\n')
                                            f.write(error_logs+'\n')
                                            f.write('#####')
                                            f.write('RAG results:\n')
                                            f.write(retrieved_texts+'\n')
                                else:
                                    retrieved_texts=RAG.RAG_search(second_test_result, collection, embed_model)
                                    if logging_:
                                        with open(logging_file_rag, 'a') as f:
                                            f.write('------------------------------')
                                            f.write('RAG input:\n')
                                            f.write(str(second_test_result)+'\n')
                                            f.write('#####')
                                            f.write('RAG results:\n')
                                            f.write(retrieved_texts+'\n')
                                second_test_result= str(second_test_result) + '\n And here is a related document. Please refer to it.' + retrieved_texts
                                if logging_:
                                    with open(logging_file, 'a') as f:
                                        f.write ('RAG results:\n')
                                        f.write(retrieved_texts+'\n')
                            if system_name=='OpenStack':
                                llm_response=chat.invoke('When I run your code, I can successfully create VM, '+ \
                                    'but VNF configuration is failed. I got this error message. Please correct the code and return the updated version.\n'+str(second_test_result))['response']
                            if system_name == 'Kubernetes':
                                llm_response=chat.invoke('When I run your code, I can successfully create Pod, '+ \
                                    'but VNF is not installed correctly as intended. Please correct the code and return the updated version by refering MOP again.\n'+str(second_test_result))['response']
                else:
                    # VM Creation failed
                    if logging_:
                        with open(logging_file, 'a') as f:
                            f.write('VM Creation test failled. Results:\n')
                            f.write(str(server_or_message)+'\n')
                    if _ < maximum_tiral-1:
                        #print(test_result)
                        #print(f'{_+2} try')
                        start_time = time.time()
                        if argparser.RAG and ('Ansible runner status:' in str(server_or_message) or 'but Pod got error: ' in str(server_or_message)):

                            # ToDo: currently consider only K8S. need to fix to consider OpenStack also.
                            error_start = check_log_error(str(server_or_message))
                            if error_start:
                                error_logs = '\n'.join(return_error_logs(str(server_or_message)))
                                retrieved_texts=RAG.RAG_search(error_logs, collection, embed_model)
                                if logging_:
                                    with open(logging_file_rag, 'a') as f:
                                        f.write('------------------------------')
                                        f.write('RAG input:\n')
                                        f.write(error_logs+'\n')
                                        f.write('#####')
                                        f.write('RAG results:\n')
                                        f.write(retrieved_texts+'\n')
                            else:
                                retrieved_texts=RAG.RAG_search(server_or_message, collection, embed_model)
                                if logging_:
                                    with open(logging_file_rag, 'a') as f:
                                        f.write('------------------------------')
                                        f.write('RAG input:\n')
                                        f.write(str(server_or_message)+'\n')
                                        f.write('#####')
                                        f.write('RAG results:\n')
                                        f.write(retrieved_texts+'\n')

                            server_or_message= str(server_or_message) + '\n And here is a related document. Please refer to it.' + retrieved_texts
                            if logging_:
                                with open(logging_file, 'a') as f:
                                    f.write ('RAG results:\n')
                                    f.write(retrieved_texts+'\n')
                        if form=='Python' and 'has no attribute ' in str(server_or_message):
                            func_name = str(server_or_message).split("'")[3]
                            #print(func_name)
                            llm_response=chat.invoke(f"Your code doesn't have '{func_name}' function. Please put the code inside it. Also, it should take 'pod_name', 'namespace', and 'image_name' as input and return True if configuration succeed. Don't use the other function name.")['response']
                        elif form=='Python' and 'positional arguments but 3 were given' in str(server_or_message):
                            llm_response=chat.invoke(f"'create_pod' function should take 'pod_name', 'namespace', and 'image_name' as input and return True if configuration succeed.")['response']
                        else:
                            llm_response=chat.invoke('When I run your code, I got this error message, and failed to create VM. Please correct the code and return the updated version.\n'+str(server_or_message))['response']
                if system_name=='OpenStack':
                    # Delete all VMs created after the target time
                    delete_vms_after(conn, target_datetime)
                elif system_name=='Kubernetes':
                    delete_all_pods(v1, apps_v1)
        # change next print operations to write in the logging_file
        if logging_:
            with open(logging_file, 'a') as f:
                f.write('Middle report\n')
                for model in model_list:
                    f.write(f"Model: {model},\n")
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
                    f.write(f"-------------------------------------------------------------\n")
        with open(logging_file_for_vnf, 'a')as f:
            f.write(f" --- MOP: {mop_file}, success num: {mop_suc_num}\n")
    
    end_time = time.time()
    #conn.compute.delete_server(floating_server.id)
    print('Total report')
    print(f"Total MOPs: {all_mop_num}")
    for model in model_list:
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
        print(f"-------------------------------------------------------------")
    with open(logging_file_for_vnf, 'a')as f:
        for vnf in success_num_by_vnf:
            f.write(f"VNF: {vnf}, Total MOPs: {success_num_by_vnf[vnf]['total_num']}, Success: {success_num_by_vnf[vnf]['success_num']}\n")
    
    print(f'Total execution time:{(end_time-total_start_time)/60/60} hours')