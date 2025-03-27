from langchain_community.llms import Ollama
from langchain.llms import OpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from secret import OPENAI_API_KEY, JUMP_HOST_IP, JUMP_HOST_PWD
from already_done import already_done
from langchain.chat_models import ChatOpenAI
import argparse
from kubernetes import client, config, stream

from docx import Document
import os
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

logging.getLogger("paramiko").setLevel(logging.CRITICAL) 

class O3MiniLLM(OpenAI):
    def _default_params(self):
        params = super()._default_params()
        if "temperature" in params:
            del params["temperature"]
        return params
if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--OpenStack', action='store_true')
    argparser.add_argument('--K8S', action='store_true')
    argparser.add_argument('--Ansible', action='store_true')
    argparser.add_argument('--Python', action='store_true')
    argparser.add_argument('--gpt', action='store_true')
    argparser.add_argument('--llama', action='store_true')
    argparser.add_argument('--rag', action='store_true')
    argparser=argparser.parse_args()
    if argparser.OpenStack:
        mop_file_path = '../mop/OpenStack_v3/'
    elif argparser.K8S:
        mop_file_path = '../mop/K8S_v1/'
    system_name='OpenStack' if argparser.OpenStack else 'Kubernetes'
    form='Python' if argparser.Python else 'Ansible'
    prompts_1, prompts_2 = prompt(form, system_name) # Prompots for each language and system.
    mop_list = [file_name for file_name in os.listdir(mop_file_path) if system_name in file_name ]
    os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
    logging_ = True
    #openai_client = OpenAI(api_key=OPENAI_API_KEY)

    maximum_tiral = 3 # Two more chance.
    if argparser.OpenStack:
        cloud_name = 'postech_cloud'
        conn = openstack.connect(cloud=cloud_name)
    elif argparser.K8S:
        config.load_kube_config()
        v1 = client.CoreV1Api()
        apps_v1 = client.AppsV1Api()
    
    logging_file = 'log.txt'
    logging_file_for_vnf = 'log_vnf.txt'
    with open(logging_file_for_vnf, 'w') as f:
        f.write('')
    model_list=[]
    if argparser.gpt:
            model_list.extend(["gpt-3.5-turbo", "gpt-4o"])
    if argparser.llama:
            model_list.extend(["llama3.3", "codellama:70b"])
    if not argparser.gpt and not argparser.llama:
        # Trying to use o3-mini, but get errors.. I think langchain version is issue, need to search.
        model_list= ["gpt-3.5-turbo", "gpt-4o", "llama3.3", #"codellama:70b", 
                     "qwen2.5-coder:32b", "qwen2:72b", "deepseek-r1:70b", "gemma3:27b", "codegemma:7b"]
    num_ctx_list = {
        "llama3.3" : 8192,
        "llama3.1:70b" : 8192,
        "qwen2" :8192,
        "qwen2:72b" : 8192,
        "gemma2" : 8192,
        "gemma2:27b" : 8192,
        "gemma3:27b" : 8192,
        "deepseek-r1:70b" : 8192,
        "codellama:70b" : 8192,
        "qwen2.5-coder:32b" : 8192,
        "codegemma:7b" : 8192
    }

    if argparser.rag:
        if argparser.OpenStack:
            db_name = 'RAG/openstack_docs.db'
        if argparser.K8S:
            db_name = 'RAG/kubernetes_docs.db'
        collection, embed_model = RAG.RAG_init(db_name)
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
    for mop_file in tqdm(mop_list[:5]):
        if mop_file in already_done:
            continue
        mop_suc_num=0
        vnf = mop_file.split('_')[1]
        lang = mop_file.split('_')[4]
        action = mop_file.split('_')[3] # confiugration action.
        if action == 'port':
            additional_action='Block the port except 22 and 80.\n'
        elif action in ['subnet', 'block'] :
            additional_action='Block the traffic except subnets with'
            if system_name == 'OpenStack':
                additional_action+=' 10.10.10.0/24 and 10.10.20.0/24.\n'
            else:
                additional_action+=' 10.244.0.0/16.\n'
            if vnf=='nDPI':
                additional_action+="Please use nDPI to block, following the MOP. Don't use the firewall.\n"
        else:
            additional_action=''
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
                llm = O3MiniLLM(model_name=model)
            else:   
                llm = Ollama(model=model, num_ctx=num_ctx_list[model])
            chat = ConversationChain(llm=llm, memory = ConversationBufferMemory())
            #llm_response=llm.invoke(prompts+mop)
            llm_response=chat.invoke(prompts_1+additional_action+prompts_2+mop)['response']
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
                    test_result, server_or_message = test_creation_python(llm_response, vnf, model, vm_num[vnf])
                else:
                    test_result, server_or_message = test_creation_ansible(llm_response, vnf, model, vm_num[vnf], v1, 600)
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
                        second_test_result = test_K8S_configuration(server_or_message, vnf, model, vm_num[vnf], v1, namespace)
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
                            if argparser.rag:
                                retrieved_texts=RAG.RAG_search(second_test_result, collection, embed_model)
                                second_test_result= str(second_test_result) + '\n And here is a related document. Please refer to it.' + retrieved_texts
                                if logging_:
                                    with open(logging_file, 'a') as f:
                                        f.write ('RAG results:\n')
                                        f.write(retrieved_texts+'\n')
                            if system_name=='OpenStack':
                                llm_response=chat.invoke('When I run your code, I can successfully create VM, '+ \
                                    'but VNF configuration is failed. I got this error message. Please fix it.\n'+str(second_test_result))['response']
                            if system_name == 'Kubernetes':
                                llm_response=chat.invoke('When I run your code, I can successfully create Pod, '+ \
                                    'but VNF is not installed correctly as intended. Please fix it.\n'+str(second_test_result))['response']
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
                        if argparser.rag and 'Ansible runner status:' in str(server_or_message):
                            retrieved_texts=RAG.RAG_search(server_or_message, collection, embed_model)
                            server_or_message= str(server_or_message) + '\n And here is a related document. Please refer to it.' + retrieved_texts
                            if logging_:
                                with open(logging_file, 'a') as f:
                                    f.write ('RAG results:\n')
                                    f.write(retrieved_texts+'\n')
                        llm_response=chat.invoke('When I run your code, I got this error message, and failed to create VM. Please fix it.\n'+str(server_or_message))['response']
            # Delete all VMs created after the target time
            if system_name=='OpenStack':
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