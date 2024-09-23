from langchain_community.llms import Ollama
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from openai import OpenAI
from secret import OPENAI_API_KEY, JUMP_HOST_IP, JUMP_HOST_PWD
from langchain.chat_models import ChatOpenAI

from docx import Document
import os
import re
import openstack
import pytz
import time
from tqdm import tqdm
import io
from datetime import datetime
import sys

def test_response(llm_response, vnf, model, vm_num, cloud_name):
    config_file_path = 'OpenStack_Conf/' 
    code_pattern = r'```python(.*?)```'
    code_pattern_second = r'```(.*?)```'
    try:
        python_code = re.findall(code_pattern, llm_response, re.DOTALL)
        if not python_code:
            python_code = re.findall(code_pattern_second, llm_response, re.DOTALL)
    except:
        #print('parsing fail')
        return "I can't see Python code in your response."
    if not python_code:
        return "I can't see Python code in your response."
    file_name = f'config_{vnf}_{model.replace(".","")}_{vm_num}.py'
    with open(config_file_path + file_name, 'w') as f:
        f.write(python_code[0])
    #print(file_name)
    try:
        create_vm = __import__(config_file_path[:-1] + '.' + file_name[:-3],fromlist=['create_vm'])
        try:
            output_capture = io.StringIO()
            sys.stdout = output_capture
            server = create_vm.create_vm()
            sys.stdout = sys.__stdout__
            captured_output = output_capture.getvalue()
            output_capture.close()
        except Exception as e:
            return e
        if server:
            print(f"VM created successfully with name: {server.name}")
            try:
                conn = openstack.connect(cloud=cloud_name)
                conn.delete_server(server.id)
                conn.delete_server(server.name)
            except:
                pass
            return True
    except Exception as e:
        #print(e)
        #print('VM creation failed')
        return e
    if captured_output:
        return captured_output
    return 'Some reason, VM creation failed.'

def delete_vms_after(conn, target_time):
    servers = conn.compute.servers(details=True)
    deleted_count = 0
    for server in servers:
        created_at = server.created_at
        created_datetime = datetime.strptime(created_at, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=pytz.UTC)
        if created_datetime > target_time:
            #print(f"Deleting VM {server.name} (Created at: {created_at})")
            conn.compute.delete_server(server.id)
            deleted_count += 1
    #print (f"Deleted {deleted_count} VMs")

if __name__ == '__main__':
    #print(llm.invoke("Tell me a joke"))
    mop_file_path = '../mop/OpenStack_v1/'
    mop_list = os.listdir(mop_file_path)
    os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
    logging = True
    #openai_client = OpenAI(api_key=OPENAI_API_KEY)

    flavor_name = 'vnf.generic.2.1024.10G'
    image_name = 'u20.04_x64'
    network_name = "'NI-data' and 'NI-management'"
    cloud_name = 'postech_cloud'
    maximum_tiral = 3 # Two more chance.
    
    prompts = 'You are an OpenStack cloud expert. ' +  \
    'Here is example code to create a VM in OpenStack using python with openstacksdk. '+ \
    "import openstack\nconn = openstack.connect(cloud='openstack_cloud')\n"+ \
    "conn.create_server('test-vm', image = 'ubuntu18.04_image', flavor = 'm1.small', network = ['net1', 'net-mgmt'])\n"+ \
    f"OpenStack server and authentication details are in config file. Cloud name is '{cloud_name}'.\n" + \
    "I'll give you a Method Of Procedure (MOP),"+ \
    'which describes the process of installing a VM in OpenStack and installing and configure the VNF on the VM. '+ \
    'With reference to this, please write the Python code that automates the process in MOP. \n' + \
    "Put the code in the function name 'create_vm' and return the 'server object' if the VM is created successfully, " + \
    "and return False if it fails. And make the part in charge of VNF configuration as a function of 'cofig_vm'. " + \
    "'config_vm' takes the 'server object' as a input and returns True if the configuration is successful, and False if it fails. "+ \
    "In this way, I hope that the same process as MOP will be performed by continuously executing the 'create_vm' function and 'config_vm'. " + \
    "Don't put usage or example \n" + \
    f"Use '{image_name}' image, '{flavor_name}' flavor, {network_name} network. \n" + \
    "If you need access to the inside of the VM for internal VM settings, instead of setting floating IP on the created VM, "+ \
    "use the Jump Host, which can connect to the internal VM, " +\
    " to connect to the newly created VM with SSH. Here is the Jump Host information. \n" + \
    f"Jump Host IP: {JUMP_HOST_IP}, Username: ubuntu, Password: {JUMP_HOST_PWD}\n" + \
    "Here is the MOP: \n" 
    
    logging_file = 'log.txt'
    model_list= ["gpt-3.5-turbo", "gpt-4o", "llama2", "llama3", "llama3.1:70b", "qwen2", "qwen2:72b", "gemma2", "gemma2:27b"]
    model_list=["gpt-4o"]
    all_mop_num = len(mop_list)
    first_create_success_num = {}
    create_success_num = {'ko':{}, 'en':{}}
    config_success_num = {} # Not Yet Implemented
    success_num_by_vnf={}
    process_time = {}
    for model in model_list:
        first_create_success_num[model] = 0
        create_success_num['ko'][model] = 0
        create_success_num['en'][model] = 0
        process_time[model] = []
    vm_num = {}
    total_start_time = time.time() 
    target_datetime = datetime.now(pytz.utc)
    for mop_file in tqdm(mop_list[:5]):
        vnf = mop_file.split('_')[1]
        lang = mop_file.split('_')[3]
        if vnf not in vm_num:
            vm_num[vnf] = 1
        else:
            vm_num[vnf] += 1
        if vnf not in success_num_by_vnf:
            success_num_by_vnf[vnf] = {'total_num' : 1, 'create_success_num': 0}
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
            if logging:
                with open(logging_file, 'a') as f:
                    f.write(f" --- Model: {model}, MOP: {mop_file}, VNF: {vnf}, VM: {vm_num[vnf]}\n")
            start_time = time.time() 
            if model.startswith('gpt'):
                llm = ChatOpenAI(temperature=0, model_name=model)
            else:   
                llm = Ollama(model=model)
            chat = ConversationChain(llm=llm, memory = ConversationBufferMemory())
            #llm_response=llm.invoke(prompts+mop)
            llm_response=chat.invoke(prompts+mop)['response']
            #print(llm_response)
            for _ in range(maximum_tiral):
                if logging:
                    with open(logging_file, 'a') as f:
                        f.write(f" --- Trial: {_+1}\n")
                        f.write(llm_response+'\n\n')
                test_result = test_response(llm_response, vnf, model, vm_num[vnf], cloud_name)
                sys.stdout = sys.__stdout__
                if test_result == True:
                    create_success_num[lang][model] += 1
                    process_time[model].append(time.time()-start_time)
                    if logging:
                        with open(logging_file, 'a') as f:
                            f.write('Test succeed\n')
                    if _>0:
                        print(f"succeed after {_+1} times trial")
                    else:
                        first_create_success_num[model] += 1
                    success_num_by_vnf[vnf]['create_success_num'] += 1
                    break
                else:
                    if _ < maximum_tiral-1:
                        #print(test_result)
                        #print(f'{_+2} try')
                        if logging:
                            with open(logging_file, 'a') as f:
                                f.write('Test failled. Results:\n')
                                f.write(str(test_result)+'\n')
                        llm_response=chat.invoke('When I run your code, I got this error message. Please fix it.\n'+str(test_result))['response']
            # Delete all VMs created after the target time
            conn = openstack.connect(cloud=cloud_name)
            delete_vms_after(conn, target_datetime)
    end_time = time.time()
    print(f"Total MOPs: {all_mop_num}")
    for model in model_list:
        print(f"Model: {model},")
        print(f"First success: {first_create_success_num[model]}")
        print(f"Korean MOP success: {create_success_num['ko'][model]}")
        print(f"English MOP success: {create_success_num['en'][model]}")
        print(f"Total Success: {create_success_num['ko'][model]+create_success_num['en'][model]}")
        if len(process_time[model]) > 0:
            print(f"Average process time: {sum(process_time[model])/len(process_time[model])} seconds")
    for vnf in success_num_by_vnf:
        print(f"VNF: {vnf}, Total MOPs: {success_num_by_vnf[vnf]['total_num']}, Success: {success_num_by_vnf[vnf]['create_success_num']}")
    print(f'Total execution time:{(end_time-total_start_time)/60} minutes')