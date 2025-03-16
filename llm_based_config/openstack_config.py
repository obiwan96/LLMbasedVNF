import openstack
import sys
import paramiko
import re
import multiprocessing
import io
from python_code_modify import wrap_code_in_main
from make_new_floating_ip import make_new_floating_ip, delete_floating_vm
from secret import JUMP_HOST_IP, JUMP_HOST_PWD
import time

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

        
def test_creation_python(llm_response, vnf, model, vm_num):
    config_file_path = 'OpenStack_Conf/'
    code_pattern = r'```python(.*?)```'
    code_pattern_second = r'```(.*?)```'
    try:
        python_code = re.findall(code_pattern, llm_response, re.DOTALL)
        if not python_code:
            python_code = re.findall(code_pattern_second, llm_response, re.DOTALL)
    except:
        #print('parsing fail')
        return False, "I can't see YAML code in your response."
    if not python_code:
        return False, "I can't see YAML code in your response."
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
        

def test_openstack_configuration(server, vnf, model, vm_num, conn, floating_server):
    jump_host_ssh = paramiko.SSHClient()
    jump_host_ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    config_file_path = 'OpenStack_Conf/'
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