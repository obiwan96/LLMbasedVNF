from langchain_community.llms import Ollama
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from secret import OPENAI_API_KEY, JUMP_HOST_IP, JUMP_HOST_PWD
from already_done import already_done
from langchain.chat_models import ChatOpenAI
import argparse

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
from make_new_floating_ip import make_new_floating_ip, delete_floating_vm

config_file_path = 'OpenStack_Conf/'
logging.getLogger("paramiko").setLevel(logging.CRITICAL) 

# Just now, preventing the infinite loop by remove top-tier code.
# I really try hard to run with time-out, but all fails. Hold on now.
preventing_loop = False

def capture_stdout(func, args=()):
    # Do not show results of LLM's code in CLI.
    stdout_capture = io.StringIO()
    sys.stdout = stdout_capture
    try:
        result = func(*args) 
    finally:
        sys.stdout = sys.__stdout__
    stdout_contents = stdout_capture.getvalue()
    stdout_capture.close()
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
            if preventing_loop:
                result, server = run_with_timeout(create_vm.create_vm, timeout = 4)
            else:
                try:
                    stdout_capture = io.StringIO()
                    sys.stdout = stdout_capture
                    server = create_vm.create_vm()
                    sys.stdout = sys.__stdout__
                    stdout_contents = stdout_capture.getvalue()
                    if server == None:
                        return False, "The 'create_vm' function does not return 'server' object."
                    return True, server
                except Exception as e:
                    sys.stdout = sys.__stdout__
                    stdout_contents = stdout_capture.getvalue()
                    stdout_capture.close()        
                    if stdout_contents:      
                        return False, stdout_contents+e
                    else:
                        return False, e
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

def wait_for_destination_ssh(ssh, destination_host, ssh_username, ssh_password, conn, floating_server):
    try:
        stdin, stdout, stderr = ssh.exec_command(f"ssh-keygen -R {destination_host}")
    except:
        return 'Error'
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
        if trial > 40:
            print('another reason, can not ssh to VM via jump host.')
            return False


def test_configuration(server, vnf, model, vm_num, conn, floating_server):
    jump_host_ssh = paramiko.SSHClient()
    jump_host_ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    while True:
        try:
            jump_host_ssh.connect(JUMP_HOST_IP, username='ubuntu', password=JUMP_HOST_PWD)
            break
        except:
            delete_floating_vm('server-vm', conn)
            make_new_floating_ip(conn)
            # Jump host VM got crashed. build again.
            #conn.compute.delete_server(floating_server.id)
            #floating_server = make_new_floating_ip(conn)
            #if not floating_server:
            #    print('Make flaoting IP failed')
            #    exit()
    try:
        vm_ip = server.addresses['NI-management'][0]['addr']
    except:
        jump_host_ssh.close()
        return "VM is created, but didn't connect to 'NI-management' network."
    # Wait until VM is avialable to SSH connection.
    while True:
        ssh_connection = wait_for_destination_ssh(jump_host_ssh,vm_ip, 'ubuntu', 'ubuntu', conn, floating_server)
        if ssh_connection == True:
            break
        elif ssh_connection == 'Error':
            jump_host_ssh.close()
            delete_floating_vm('server-vm', conn)
            make_new_floating_ip(conn)
            jump_host_ssh.connect(JUMP_HOST_IP, username='ubuntu', password=JUMP_HOST_PWD)
        else: 
            jump_host_ssh.close()
            return 'Error: Can not SSH to VM with jump host.'
    file_name = f'config_{vnf}_{model.replace(".","")}_{vm_num}.py'
    try:
        config_vm = __import__(config_file_path[:-1] + '.' + file_name[:-3],fromlist=['config_vm'])
        try:
            if preventing_loop:
                time_out_result, result = run_with_timeout(config_vm.config_vm, (server,), 7)
            else:
                try:
                    stdout_capture = io.StringIO()
                    sys.stdout = stdout_capture
                    result = config_vm.config_vm(server)
                    sys.stdout = sys.__stdout__
                    stdout_contents = stdout_capture.getvalue()
                    if result == None:
                        return 'config_vm function does not return anything. It should return the whether the configuration is successful or not.'
                    time_out_result = True                  
                except Exception as e:
                    sys.stdout = sys.__stdout__
                    stdout_contents = stdout_capture.getvalue()
                    stdout_capture.close()        
                    if stdout_contents:      
                        return stdout_contents+e
                    else:
                        return e
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
            if result == False:
                if 'stdout_contents' in locals():
                    return 'Your code return the False. It seems to fail to configure, and here are outputs: \n'+stdout_contents
                return 'Your code return the False. It seems to fail to configure.'
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
                pass
                #print(f"Deleting VM {server.name} (Created at: {created_at})")
            conn.compute.delete_server(server.id)
            deleted_count += 1
        #else:
        #    print(f"VM (Created at: {created_at}), name : {server.name} is not deleted")
    if logging_:
        pass
        #print (f"Deleted {deleted_count} VMs")

def read_good_example(method, platform, example_path = 'Good_Example/'):
    good_example = {}
    for file in os.listdir(example_path):
        if platform in file and file.endswith(method):
            with open(example_path+file, 'r') as f:
                good_example[file] = f.read()
    return good_example

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--OpenStack', action='store_true')
    argparser.add_argument('--K8S', action='store_true')
    argparser.add_argument('--Ansible', action='store_true')
    argparser.add_argument('--Python', action='store_true')
    argparser.add_argument('--gpt', action='store_true')
    aprgparser.add_argument('--llama', action='store_true')
    #print(llm.invoke("Tell me a joke"))
    if argparser.OpenStack:
        mop_file_path = '../mop/OpenStack_v3/'
    elif argparser.K8S:
        mop_file_path = '../mop/K8S/'
    mop_list = os.listdir(mop_file_path)
    os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
    logging_ = True
    #openai_client = OpenAI(api_key=OPENAI_API_KEY)

    flavor_name = 'vnf.generic.2.1024.10G'
    image_name = 'u20.04_x64'
    network_name = "'NI-data' and 'NI-management'"
    cloud_name = 'postech_cloud'
    maximum_tiral = 3 # Two more chance.
    if argparser.OpenStack:
        conn = openstack.connect(cloud=cloud_name)
    
    if argparser.Python:
        if argparser.OpenStack:
            good_example= read_good_example('py', 'OpenStack')
        elif argparser.K8S:
            good_example= read_good_example('py', 'k8s')
        else:
            good_example = read_good_example('py')
    elif argparser.Ansible:
        if argparser.OpenStack:
            good_example= read_good_example('yml', 'OpenStack')
        elif argparser.K8S:
            good_example= read_good_example('yml', 'k8s')
        else:
            good_example = read_good_example('yml')
    else:
        print('Currently, only supporting Python and Ansible. Specify which one you use.')
        sys.exit(0)
    good_example_str=''
    for example in good_example:
        good_example_str += example+':\n'+good_example[example] +'\n'
    if argparser.Python:
        if argparser.OpenStack:
            prompts_1 = '''You are an OpenStack cloud expert. 
            Here are 'example' codes to create and configurate a VNF in OpenStack using python with openstacksdk. \n'''
            good_example_str + \
            f'''\nPlease rememeber, these are example code, so you have to just refer to them. For detailed VNF setup methods and parameters, follow the following description, not the example code.
            OpenStack server and authentication details are in config file. Cloud name is '{cloud_name}'.
            I'll give you a Method Of Procedure (MOP),"
            which describes the process of installing a VM in OpenStack and installing and configure the VNF on the VM. 
            With reference to this, please write the Python code that automates the process in MOP. 
            Put the code to create VM in the function name 'create_vm' and return the 'server object' if the VM is created successfully, 
            and return False if it fails. And make the part in charge of VNF configuration as a function of 'cofig_vm'.
            'config_vm' takes the 'server object' as a input and returns True if the configuration is successful, and False if it fails. 
            In this way, I hope that the same process as MOP will be performed by continuously executing the 'create_vm' function and 'config_vm'.'''
        
            prompts_2= f'''Don't seperate two fucntions, put in same code block, and don't put usage or example in the block.
            Use '{image_name}' image, '{flavor_name}' flavor, {network_name} network.
            Don't make key pair in OpenStack, don't use stdin to get any kind of password.
            If you need access to the inside of the VM for internal VM settings, instead of setting floating IP on the created VM,
            use the Jump Host, which can connect to the internal VM, 
            to connect to the newly created VM with SSH. Here is the Jump Host information.
            You will need to enable SSH connection in newly created VM through password to enable connection from Jump Host,
            and you will need to set an ID and password to 'ubuntu'. You should get an IP address in 'server' object and use it to connect in VM.
            Jump Host IP: {JUMP_HOST_IP}, Username: ubuntu, Password: {JUMP_HOST_PWD}
            When if you need to modify some files in VM, Paramiko is not an interactive session, so you should not use vim or nano. I recommend to use echo, but you can find other way.
            Every time you access the VM with Paramiko, it connect to '/home' directory, so 'cd' does not work. I recommend you to use the absolute path.
            Here is the MOP: '''

        # K8S Python code part is just copy version of OpenStack. need change.
        elif argparser.K8S:
            prompts_1 = '''You are an Kubernetes cloud expert. 
            Here are 'example' codes to create and configurate a CNF in Kubernetes using Python with Kubernetes library. \n'''
            good_example_str + \
            f'''\nPlease rememeber, these are example code, so you have to just refer to them. For detailed CNF setup methods and parameters, follow the following description, not the example code.
            The Kubernetes configuration file path is '/home/dpnm/.kube/config'.
            I'll give you a Method Of Procedure (MOP),"
            which describes the process of installing a Pod in Kubernetes and installing and configure the CNF on the Pod. 
            With reference to this, please write the Python code that automates the process in MOP. 
            Put the code to create Pod in the function name 'create_Pod' and return the 'server object' if the Pod is created successfully, 
            and return False if it fails. And make the part in charge of CNF configuration as a function of 'cofig_Pod'.
            'config_Pod' takes the 'server object' as a input and returns True if the configuration is successful, and False if it fails. 
            In this way, I hope that the same process as MOP will be performed by continuously executing the 'create_Pod' function and 'config_Pod'.'''
        
            prompts_2= f'''Don't seperate two fucntions, put in same code block, and don't put usage or example in the block.
            Use '{image_name}' image, '{flavor_name}' flavor, {network_name} network.
            Don't make key pair in Kubernetes, don't use stdin to get any kind of password.
            If you need access to the inside of the Pod for internal Pod settings, instead of setting floating IP on the created Pod,
            use the Jump Host, which can connect to the internal Pod, 
            to connect to the newly created Pod with SSH. Here is the Jump Host information.
            You will need to enable SSH connection in newly created Pod through password to enable connection from Jump Host,
            and you will need to set an ID and password to 'ubuntu'. You should get an IP address in 'server' object and use it to connect in Pod.
            Jump Host IP: {JUMP_HOST_IP}, Username: ubuntu, Password: {JUMP_HOST_PWD}
            When if you need to modify some files in Pod, Paramiko is not an interactive session, so you should not use vim or nano. I recommend to use echo, but you can find other way.
            Every time you access the Pod with Paramiko, it connect to '/home' directory, so 'cd' does not work. I recommend you to use the absolute path.
            Here is the MOP: '''

    elif argparser.Ansible:
        if argparser.OpenStack:
            continue
        # Let's do K8S Ansible first!
        elif argparser.K8S:
            prompts_1 = '''You are an Kubernetes cloud expert. 
            Here are 'example' configuration codes by YAML to create and configurate a CNF in Kubernetes using Ansible. \n'''
            good_example_str + \
            f'''\nPlease rememeber, these are YAML code, so you have to just refer to them. 
            For detailed CNF setup methods and parameters, follow the following description, not the example YAML code.
            The Kubernetes configuration file path is '/home/dpnm/.kube/config'.
            I'll give you a Method Of Procedure (MOP),"
            which describes the process of installing a Pod in Kubernetes and installing and configure the CNF on the Pod. 
            With reference to this, please write the YAML code that automates the process in MOP. '''
        
            prompts_2= f'''Use '{image_name}' image, '{flavor_name}' flavor, {network_name} network.
            Here is the MOP: '''
    
    logging_file = 'log.txt'
    logging_file_for_vnf = 'log_vnf.txt'
    with open(logging_file_for_vnf, 'w') as f:
        f.write('')
    if argparser.gpt:
            model_list= ["gpt-3.5-turbo", "gpt-4o"]
    elif argparser.llama:
            model_list= ["llama3", "llama3.1:70b"]
    else:
        # Needs some version changes. 
        # Need to check steel they are latest one.
        # Need to add some codes-specific LLM.
        # Maybe add 1 or 2 recently developed LLM.

        model_list= ["gpt-3.5-turbo", "gpt-4o", "llama3", "llama3.1:70b", "qwen2", "qwen2:72b", "gemma2", "gemma2:27b"]
    num_ctx_list = {
        "llama3" : 8192,
        "llama3.1:70b" : 8192,
        "qwen2" :8192,
        "qwen2:72b" : 8192,
        "gemma2" : 8192,
        "gemma2:27b" : 8192
    }
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
        if mop_file in already_done:
            continue
        mop_suc_num=0
        vnf = mop_file.split('_')[1]
        lang = mop_file.split('_')[4]
        action = mop_file.split('_')[3] # confiugration action.
        if action == 'port':
            additional_action='Block the port except 22 and 80.\n'
        elif action in ['subnet', 'block'] :
            additional_action='Block the traffic except subnets with 10.10.10.x and 10.10.20.x\n'
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
                test_result, server_or_message = test_creation(llm_response, vnf, model, vm_num[vnf])
                spend_time[1] = time.time()-start_time
                if test_result == True:
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

                    second_test_result = test_configuration(server_or_message, vnf, model, vm_num[vnf], conn, None)
                    spend_time[2] = time.time()-start_time
                    conn.delete_server(server_or_message.id)
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
                        start_time = time.time()
                        llm_response=chat.invoke('When I run your code, I got this error message, and failed to create VM. Please fix it.\n'+str(server_or_message))['response']
            # Delete all VMs created after the target time
            delete_vms_after(conn, target_datetime)
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