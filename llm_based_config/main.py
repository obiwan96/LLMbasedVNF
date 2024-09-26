from langchain_community.llms import Ollama
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
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
import paramiko
import logging
import time
import multiprocessing

from python_code_modify import wrap_code_in_main


config_file_path = 'OpenStack_Conf/'
logging.getLogger("paramiko").setLevel(logging.CRITICAL) 

def capture_stdout(func, args=()):
    # Do not show results of LLM's code in CLI.
    stdout_capture = io.StringIO()
    sys.stdout = stdout_capture
    try:
        result = func(*args) 
    finally:
        sys.stdout = sys.__stdout__
    stdout_contents = stdout_capture.getvalue()
    return stdout_contents, result

def run_with_timeout(func, args=(), timeout = 5):
    # Run the function with a timeout
    with multiprocessing.Pool(processes=1) as pool:
        result = pool.apply_async(capture_stdout, args=(func,args,))
        try:
            stdout, return_value = result.get(timeout=timeout*60) # Timeout in minutes
            return True, return_value
        except multiprocessing.TimeoutError:
            if 'stdout' in locals():
                return False, stdout+f"Timeout reached. Function did not finish within {timeout} minutes."
            return False, f"Timeout reached. Function did not finish within {timeout} minutes."

        
def test_creation(llm_response, vnf, model, vm_num):
    code_pattern = r'```python(.*?)```'
    code_pattern_second = r'```(.*?)```'
    try:
        python_code = re.findall(code_pattern, llm_response, re.DOTALL)
        if not python_code:
            python_code = re.findall(code_pattern_second, llm_response, re.DOTALL)
    except:
        #print('parsing fail')
        return False, "I can't see Python code in your response."
    if not python_code:
        return False, "I can't see Python code in your response."
    file_name = f'config_{vnf}_{model.replace(".","")}_{vm_num}.py'
    with open(config_file_path + file_name, 'w') as f:
        f.write(python_code[0])
    result = wrap_code_in_main(config_file_path + file_name, config_file_path + file_name)
    if not result:
        return False, 'Code parsing failed. Maybe some syntax error or unexpected indentation occured.'
    #print(file_name)
    try:
        create_vm = __import__(config_file_path[:-1] + '.' + file_name[:-3],fromlist=['create_vm'])
        try:
            result, server = run_with_timeout(create_vm.create_vm, timeout = 4)
        except Exception as e:
            return False, e
        if result:
            #print(f"VM created successfully with name: {server.name}")
            '''try:
                conn = openstack.connect(cloud=cloud_name)
                conn.delete_server(server.id)
            except:
                pass'''
            return True, server
        else:
            return False, server
    except Exception as e:
        #print(e)
        #print('VM creation failed')
        return False, e

def vm_ssh_config_check(vm_ssh, input, output, exactly=False):
    stdin, stdout, stderr = vm_ssh.exec_command(input)
    if exactly:
        msg = stdout.read().decode("utf-8").strip()
        if msg == output:
            return True
    else:
        msg = stdout.read().decode("utf-8")
        if output in msg:
            return True
    return False

def wait_for_destination_ssh(ssh, destination_host, ssh_username, ssh_password):
    stdin, stdout, stderr = ssh.exec_command(f"ssh-keygen -R {destination_host}")
    trial=0
    while True:
        trial+=1
        try:
            # Create an SSH session from the jump host to the destination server
            jump_transport = ssh.get_transport()
            dest_addr = (destination_host, 22)
            local_addr = (JUMP_HOST_IP, 22)

            # Open a direct TCP channel to the destination server
            jump_channel = jump_transport.open_channel("direct-tcpip", dest_addr, local_addr)
            
            # Create a new SSH client to connect to the destination through the jump host
            ssh_dest = paramiko.SSHClient()
            ssh_dest.set_missing_host_key_policy(paramiko.AutoAddPolicy())

            # Try connecting to the destination server
            ssh_dest.connect(destination_host, username=ssh_username, password=ssh_password, sock=jump_channel)
            ssh_dest.close()
            return True
        except:
            time.sleep(5)
        if trial > 35:
            print('another reason, can not ssh to VM via jump host.')
            return False


def test_configuration(server, vnf, model, vm_num):
    jump_host_ssh = paramiko.SSHClient()
    jump_host_ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    jump_host_ssh.connect(JUMP_HOST_IP, username='ubuntu', password=JUMP_HOST_PWD)
    try:
        vm_ip = server.addresses['NI-management'][0]['addr']
    except:
        jump_host_ssh.close()
        return "VM is created, but didn't connect to 'NI-management' network."
    # Wait until VM is avialable to SSH connection.
    if not wait_for_destination_ssh(jump_host_ssh,vm_ip, 'ubuntu', 'ubuntu'):
        jump_host_ssh.close()
        return 'Error: Can not SSH to VM with jump host.'
    file_name = f'config_{vnf}_{model.replace(".","")}_{vm_num}.py'
    try:
        config_vm = __import__(config_file_path[:-1] + '.' + file_name[:-3],fromlist=['config_vm'])
        try:
            time_out_result, result = run_with_timeout(config_vm.config_vm, (server,), 7)
        except Exception as e:
            jump_host_ssh.close()
            return e
        if time_out_result and result == True:
            vm_ssh = paramiko.SSHClient()
            vm_ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            jump_host_transport = jump_host_ssh.get_transport()
            dest_addr = (vm_ip, 22)
            local_addr = ('127.0.0.1', 22)
            vm_channel = jump_host_transport.open_channel("direct-tcpip", dest_addr, local_addr)
            vm_ssh.connect(vm_ip, username='ubuntu', password= 'ubuntu', sock=vm_channel)

           ############################################
           # Simply check if the VNF is running well. #
           # Somedays, need to check more detailly.   #
           ############################################
            if vnf == 'firewall':
                if vm_ssh_config_check(vm_ssh, 'sudo iptables -L -v -n', 'DROP'):
                    vm_ssh.close()
                    jump_host_ssh.close()
                    return True
            elif vnf == 'Haproxy':
                if vm_ssh_config_check(vm_ssh, 'systemctl is-active haproxy', 'active', exactly=True):
                    if vm_ssh_config_check(vm_ssh, 'haproxy -c -f /etc/haproxy/haproxy.cfg', 'Configuration file is valid'):
                        vm_ssh.close()
                        jump_host_ssh.close()
                        return True
            elif vnf == 'nDPI':
                if vm_ssh_config_check(vm_ssh, 'ps aux', 'ndpiReader'):
                    vm_ssh.close()
                    jump_host_ssh.close()
                    return True
            elif vnf == 'ntopng':
                if vm_ssh_config_check(vm_ssh, 'ps aux', 'ntopng'):
                    vm_ssh.close()
                    jump_host_ssh.close()
                    return True
            elif vnf == 'Suricata':
                if vm_ssh_config_check(vm_ssh, 'systemctl is-active suricata', 'active', exactly=True):
                    vm_ssh.close()
                    jump_host_ssh.close()
                    return True
            else:
                print('weried...')
            vm_ssh.close()
            jump_host_ssh.close()
            # Close SSH connections
            return 'Your code is run well, but when I check the VM, VNF is not installed correctly as intended.'
        else:
            jump_host_ssh.close()
            return result
    except Exception as e:
        #print(e)
        #print('VM creation failed')
        if 'result' in locals():
            vm_ssh.close()
        jump_host_ssh.close()
        return e

def delete_vms_after(conn, target_time, logging_=False):
    servers = conn.compute.servers(details=True)
    #print(f'target time: {target_time}')
    deleted_count = 0
    for server in servers:
        created_at = server.created_at
        created_datetime = datetime.strptime(created_at, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=pytz.UTC)
        if created_datetime > target_time:
            if logging_:
                print(f"Deleting VM {server.name} (Created at: {created_at})")
            conn.compute.delete_server(server.id)
            deleted_count += 1
        #else:
        #    print(f"VM (Created at: {created_at}), name : {server.name} is not deleted")
    if logging_:
        print (f"Deleted {deleted_count} VMs")

def read_good_example(example_path = 'Good_Example/'):
    good_example = {}
    for file in os.listdir(example_path):
        with open(example_path+file, 'r') as f:
            good_example[file] = f.read()
    return good_example

if __name__ == '__main__':
    #print(llm.invoke("Tell me a joke"))
    mop_file_path = '../mop/OpenStack_v1/'
    mop_list = os.listdir(mop_file_path)
    os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
    logging_ = True
    #openai_client = OpenAI(api_key=OPENAI_API_KEY)

    flavor_name = 'vnf.generic.2.1024.10G'
    image_name = 'u20.04_x64'
    network_name = "'NI-data' and 'NI-management'"
    cloud_name = 'postech_cloud'
    maximum_tiral = 3 # Two more chance.
    conn = openstack.connect(cloud=cloud_name)
    
    good_example = read_good_example()
    good_example_str=''
    for example in good_example:
        good_example_str += example+'\n'+good_example[example]
    prompts = 'You are an OpenStack cloud expert. ' +  \
    'Here is example code to create and configurate a VM in OpenStack using python with openstacksdk. \n'+ \
    good_example_str + \
    f"OpenStack server and authentication details are in config file. Cloud name is '{cloud_name}'.\n" + \
    "I'll give you a Method Of Procedure (MOP),"+ \
    'which describes the process of installing a VM in OpenStack and installing and configure the VNF on the VM. '+ \
    'With reference to this, please write the Python code that automates the process in MOP. \n' + \
    "Put the code in the function name 'create_vm' and return the 'server object' if the VM is created successfully, " + \
    "and return False if it fails. And make the part in charge of VNF configuration as a function of 'cofig_vm'. " + \
    "'config_vm' takes the 'server object' as a input and returns True if the configuration is successful, and False if it fails. "+ \
    "In this way, I hope that the same process as MOP will be performed by continuously executing the 'create_vm' function and 'config_vm'. " + \
    "Don't seperate two fucntions, put in same code block, and don't put usage or example in the block.\n" + \
    f"Use '{image_name}' image, '{flavor_name}' flavor, {network_name} network. \n" + \
    "Don't make key pair in OpenStack, don't use stdin to get any kind of password. \n"
    "If you need access to the inside of the VM for internal VM settings, instead of setting floating IP on the created VM, "+ \
    "use the Jump Host, which can connect to the internal VM, " +\
    " to connect to the newly created VM with SSH. Here is the Jump Host information." + \
    "You will need to enable SSH connection through password to enable connection to the target VM," + \
    " and you will need to set ID and password to 'ubuntu'.\n"
    f"Jump Host IP: {JUMP_HOST_IP}, Username: ubuntu, Password: {JUMP_HOST_PWD}\n" + \
    "Here is the MOP: \n" 
    
    logging_file = 'log.txt'
    logging_file_for_vnf = 'log_vnf.txt'
    with open(logging_file, 'w') as f:
        f.write('')
    model_list= ["gpt-3.5-turbo", "gpt-4o", "llama2", "llama3", "llama3.1:70b", "qwen2", "qwen2:72b", "gemma2", "gemma2:27b"]
    #model_list= ["gpt-4o", "llama3"]
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
    for mop_file in tqdm(mop_list):
        mop_suc_num=0
        vnf = mop_file.split('_')[1]
        lang = mop_file.split('_')[3]
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
            start_time = time.time() 
            if model.startswith('gpt'):
                llm = ChatOpenAI(temperature=0, model_name=model)
            else:   
                llm = Ollama(model=model)
            chat = ConversationChain(llm=llm, memory = ConversationBufferMemory())
            #llm_response=llm.invoke(prompts+mop)
            llm_response=chat.invoke(prompts+mop)['response']
            #print(llm_response)
            already_success = False
            for _ in range(maximum_tiral):
                if logging_:
                    with open(logging_file, 'a') as f:
                        f.write(f" --- Trial: {_+1}\n")
                        f.write(llm_response+'\n\n')
                test_result, server_or_message = test_creation(llm_response, vnf, model, vm_num[vnf])
                if test_result == True:
                    try:
                        server = conn.compute.wait_for_server(server_or_message)
                    except:
                        error_message="The 'create_vm' function does not return 'server' object. "
                        if logging_:
                            with open(logging_file, 'a') as f:
                                f.write(error_message+'\n')
                        if _ < maximum_tiral-1:
                            llm_response=chat.invoke(error_message+'Please fix it.\n')['response']
                        continue
                    # VM creation success.
                    # Now, test the configuration
                    if not already_success:
                        create_success_num[lang][model] += 1
                        already_success = True
                    #if _>0:
                    #    print(f"succeed after {_+1} times trial")
                    if _ == 0:
                        first_create_success_num[model] += 1
                    
                    second_test_result = test_configuration(server_or_message, vnf, model, vm_num[vnf])
                    conn.delete_server(server_or_message.id)
                    if second_test_result == True:
                        process_time[model].append(time.time()-start_time)
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
                        if logging_:
                            with open(logging_file, 'a') as f:
                                f.write('Config test failled. Results:\n')
                                f.write(str(second_test_result)+'\n')
                        if _ < maximum_tiral-1:
                            llm_response=chat.invoke('When I run your code, I can successfully create VM, '+ \
                                'but VNF configuration is failed. I got this error message. Please fix it.\n'+str(second_test_result))['response']
                else:
                    # VM Creation failed
                    if logging_:
                        with open(logging_file, 'a') as f:
                            f.write('VM Creation test failled. Results:\n')
                            f.write(str(server_or_message)+'\n')
                    if _ < maximum_tiral-1:
                        #print(test_result)
                        #print(f'{_+2} try')
                        llm_response=chat.invoke('When I run your code, I got this error message, and failed to create VM. Please fix it.\n'+str(server_or_message))['response']
            # Delete all VMs created after the target time
            delete_vms_after(conn, target_datetime)
        print('Middle report')
        for model in model_list:
            print(f"Model: {model},")
            print(f"First VM Create success: {first_create_success_num[model]}")
            print(f"Total VM Create Success: {create_success_num['ko'][model]+create_success_num['en'][model]}")
            print(f"First VNF Config success: {first_config_success_num[model]}")
            print(f"Total VNF Config Success: {config_success_num['ko'][model]+config_success_num['en'][model]}")
            print(f"-------------------------------------------------------------")
        with open(logging_file_for_vnf, 'a')as f:
            f.write(f" --- MOP: {mop_file}, success num: {mop_suc_num}\n")
    
    end_time = time.time()
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
            print(f"Average process time: {sum(process_time[model])/len(process_time[model])} seconds")
        print(f"-------------------------------------------------------------")
    for vnf in success_num_by_vnf:
        print(f"VNF: {vnf}, Total MOPs: {success_num_by_vnf[vnf]['total_num']}, Success: {success_num_by_vnf[vnf]['success_num']}")
    print(f'Total execution time:{(end_time-total_start_time)/60/60} hours')