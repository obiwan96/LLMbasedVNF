import ansible_runner
from kubernetes import client, config, stream
from kubernetes.client.rest import ApiException
import yaml
from python_code_modify import wrap_code_in_main
import io
import re
import sys
from prompt import namespace
import time
import os
import subprocess

image_name='dokken/ubuntu-20.04'

def list_pods_in_namespace(v1, namespace=namespace):
    pods = v1.list_namespaced_pod(namespace)
    
    return [pod.metadata.name for pod in pods.items]

def update_yml(data):
    if isinstance(data, dict):
        for key, value in data.items():
            if key == 'namespace':
                data[key] = '{{ namespace }}'
            if key == 'definition' and 'metadata' in value:
                if 'name' in data[key]['metadata'].keys():
                    data[key]['metadata']['name'] = '{{ pod_name }}'
                if 'namespace' in data[key]['metadata'].keys():
                    data[key]['metadata']['namespace'] = '{{ namespace }}'
            elif key == 'hostNetwork':
                # hostNetwork should be false!!
                # If not, Pod's network confiugration effect to Node's
                data[key] = 'false'
            else:
                update_yml(value)
    elif isinstance(data, list):
        for item in data:
            update_yml(item)

def check_yml_if_do_in_localhost_or_nodes(data):
    for play in data:
        if "hosts" in play:
            if str(play["hosts"]).strip().lower() == "localhost":
                if "tasks" in play:
                    for task in play["tasks"]:
                        if not 'k8s' in str(task) and not 'kubernetes' in str(task):
                            # Suspect. It may do something in localhost.
                            return False
            if str(play["hosts"]).strip().lower() == "all":
                return False
    return True

def ansi_result_clear(ansible_output):
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    clean_output = ansi_escape.sub('', ansible_output)
    return clean_output

class OutputCollector:
    def __init__(self):
        self.captured_output = ''

    def event_handler(self, event_data):
        if 'stdout' in event_data and event_data['stdout']:
            self.captured_output+=event_data['stdout']

def get_pod_logs(v1, pod_name, namespace= namespace):
    try:
        logs = v1.read_namespaced_pod_log(name=pod_name, namespace=namespace)
        return logs
    
    except ApiException as e:
        print(f"Exception when reading logs in pod {pod_name}: {e}")

def check_log_error(logs):
    error_words=['error ', 'error:', 'error!']
    for error_word in error_words:
        if error_word in logs.lower():
            return logs.lower().find(error_word)
        
    if 'ERR' in logs:
        return logs.find('ERR')
    return False

def test_creation_ansible_K8S(llm_response, vnf, model, vm_num, v1, timeout=300):
    config_file_path = 'K8S_Conf/'
    if not os.path.exists(config_file_path):
        os.makedirs(config_file_path)
    code_pattern_list = [r'```yaml(.*?)```', r'```(.*?)```', r'-{10,}\n(.*?)\n-{10,}', r'^---\n(.*)$']

    try:
        for code_pattern in code_pattern_list:
            yml_code = re.findall(code_pattern, llm_response, re.DOTALL)
            if yml_code:
                break
    except:
        #print('parsing fail')
        return False, "I can't see YAML code in your response."
    if not yml_code:
        return False, "I can't see YAML code in your response."
    try:
        pod_name= vnf.lower()+'-pod'
        yaml_data = yaml.safe_load(yml_code[0])
        update_yml(yaml_data)
        if not check_yml_if_do_in_localhost_or_nodes(yaml_data):
            return False, f'''Your YAML code runs on localhost or Kubernetes nodes, but there's no task to perform there. Please update it to run only inside Kubernetes.'''
        #if not '{{ pod_name }}' in str(yaml_data) or not '{{ namespace }}' in str(yaml_data):
        #    return False, f"YAML parsing failed. Please use {pod_name} as pod name and {namespace} as namespace in the YAML code by refering the example code."
    except:
        return False, "YAML parsing failed. Please check the format of the YAML code. Please refer to the example code again."
    file_name = f'config_{vnf}_{model.replace(".","")}_{vm_num}.yml'
    with open(config_file_path + file_name, 'w') as f:
        yaml.dump(yaml_data, f)
    #print(file_name)
    try:
        start_time = time.time()
        collector = OutputCollector()
        if not check_pod_errors(v1, 'kube-system') or not check_pod_errors(v1, 'kube-flannel'):
            restart_daemons(v1)
        runner_thread, runner = ansible_runner.run_async(
            private_data_dir=config_file_path, playbook=file_name,
            extravars = {'pod_name': pod_name, 'namespace' : namespace, 'image': image_name, 'image_name': image_name},
            quiet = True, # no stdout.
            event_handler=collector.event_handler
            )
        while runner_thread.is_alive():
            if time.time() - start_time > timeout:
                return False, f"Ansible running doesn't end within {timeout} seconds. Test fail.\n Here are Ansible running results:\n"+collector.captured_output
            time.sleep(3)

        #print("상태:", response.status)
        #print("반환 코드:", response.rc)
        
        if runner.rc == 0:
            if 'skipping: no hosts matched' in collector.captured_output:
                return False, 'There was a task that was skipped due to an incorrect host name. Please correct the host name.'
            # Ansible run well
            try:
                # Wait for creation success for Pod.                
                while True:
                    try:
                        pod = v1.read_namespaced_pod(name=pod_name, namespace=namespace)
                    except client.exceptions.ApiException as e:
                        pod = None
                    if pod:
                        phase = pod.status.phase
                        #container_statuses = pod.status.container_statuses or []
                        # check if all container are in ready status
                        
                        ready = True
                        # let's start from not for all container now.

                        #ready = all(cs.ready for cs in container_statuses) if container_statuses else False
                        
                        if phase == "Running" and ready:
                            # Pod is creating successfully
                            return True, pod_name
                        elif phase in ["CrashLoopBackOff", "Error"]:
                            # Pod is in error state.
                            error_logs = get_pod_logs(v1, pod_name, namespace)
                            if error_logs:
                                return False, "Ansible ran, but Pod got error: "+ error_logs
                            else:
                                return False, f"Ansible ran, but Pod got into {phase} status."
                    if time.time() - start_time > timeout:
                        return False, f"Ansible ran succeed, but the containers are not ready whitin {timeout} seconds. Test fail."
                    time.sleep(5)
            except ApiException as e:
                if e.status == 404:
                    return False, f"Ansible ran succeeed. But Pod '{pod_name}' doesn't exist in the Kubernetes namespace '{namespace}."
                else:
                    return False, "Error while searching Pod"
            
        else:
            ansible_output=ansi_result_clear(collector.captured_output)
            return False, 'Ansible ran failed. Ansible runner status: '+runner.status + '\n Here are Ansible running results:\n'+ansible_output

    except Exception as e:
        #print(e)
        #print('VM creation failed')
        return False, e
    
def test_creation_python_K8S(llm_response, vnf, model, vm_num, trial, v1, namespace, timeout=300):
    config_file_path = 'K8S_Conf/'
    if not os.path.exists(config_file_path):
        os.makedirs(config_file_path)
    code_pattern_list = [r'```python(.*?)```', r'```(.*?)```', r'-{10,}\n(.*?)\n-{10,}', r'^---\n(.*)$']

    try:
        for code_pattern in code_pattern_list:
            python_code = re.findall(code_pattern, llm_response, re.DOTALL)
            if python_code:
                break
    except:
        #print('parsing fail')
        return False, "I can't see Python code in your response."
    if not python_code:
        return False, "I can't see Python code in your response."
    
    file_name = f'config_{vnf}_{model.replace(".","")}_{vm_num}_{trial}.py'
    with open(config_file_path + file_name, 'w') as f:
        f.write(python_code[0])
    result = wrap_code_in_main(config_file_path + file_name, config_file_path + file_name)
    if not result:
        return False, 'Code parsing failed. Maybe some syntax error or unexpected indentation occured.'
   
    try:
        create_pod = __import__(config_file_path[:-1] + '.' + file_name[:-3],fromlist=['create_pod'])
    except Exception as e:
        return False, str(e)+" Please put the code inside the 'create_pod' function."
    if not hasattr(create_pod, 'create_pod'):
        #print (config_file_path[:-1] + '.' + file_name[:-3])
        if 'create_pod(' in python_code[0]:
            return False, r"I got error while import 'create_pod'. Please check the indentation or grammar."
        return False, " Please put the code inside the 'create_pod' function."

    try:
        if not check_pod_errors(v1, 'kube-system') or not check_pod_errors(v1, 'kube-flannel'):
            restart_daemons(v1)
        start_time = time.time()
        stdout_capture = io.StringIO()
        sys.stdout = stdout_capture
        pod_name= vnf.lower()+'-pod'
        pod_creation_result = create_pod.create_pod(pod_name, namespace, image_name)
        sys.stdout = sys.__stdout__
        stdout_contents = stdout_capture.getvalue()
        if pod_creation_result:
            # The pod_creation function ran succeed.
            # Wait for creation success for Pod.    
            try:            
                while True:
                    try:
                        pod = v1.read_namespaced_pod(name=pod_name, namespace=namespace)
                    except client.exceptions.ApiException as e:
                        pod = None
                    if pod:
                        phase = pod.status.phase                        
                        #ready = True
                        
                        if phase == "Running":
                            # Pod is creating successfully
                            return True, pod_name
                        elif phase in ["CrashLoopBackOff", "Error"]:
                            # Pod is in error state.
                            error_logs = get_pod_logs(v1, pod_name, namespace)
                            if error_logs:
                                return False, "'create_pod' ran, but Pod got error: "+ error_logs
                            else:
                                return False, f"'create_pod' ran, but Pod got into {phase} status."
                    if time.time() - start_time > timeout:
                        return False, f"'create_pod' ran succeed, but the containers are not ready whitin {timeout} seconds. Test fail. Please check again you use right image_name, namespace variables for pod."
                    time.sleep(5)
            except ApiException as e:
                if e.status == 404:
                    return False, f"'create_pod' ran succeeed. But Pod '{pod_name}' doesn't exist in the Kubernetes namespace '{namespace}."
                else:
                    return False, "Error while searching Pod"
        else:
            return False, "'create_pod' didn't return True. Here is the output. \n"+stdout_contents
    except Exception as e:
        sys.stdout = sys.__stdout__
        stdout_contents = stdout_capture.getvalue()
        stdout_capture.close()        
        if stdout_contents:      
            return False, stdout_contents+str(e)
        else:
            return False, e
    #except Exception as e:
    #    return False, str(e)+" Please put the code inside the 'create_pod' function."

def run_config(v1, pod_name, namespace, input, output, exactly=False):
    input = input.split()
    try:
        response = stream.stream(
            v1.connect_get_namespaced_pod_exec,
            pod_name,
            namespace,
            command=input,
            stderr=True,
            stdin=False,
            stdout=True,
            tty=False,
            _preload_content=True
        )
        #print("명령어 실행 결과:")
        #print(response)
    except ApiException as e:
        return str(e)
    #print( response)
    if exactly:
        #mresponsesg = response.read().decode("utf-8").strip()
        if response == output:
            return True
    else:
        #response = response.read().decode("utf-8")
        if output in response:
            return True
    return response

def test_K8S_configuration(pod_name, vnf, v1, namespace, wait_time=150):
    # I can't find how to check all commands run well.
    # Just waiting is seems better. 150 is enough?
    start_time = time.time()
    while True:
        pod = v1.read_namespaced_pod(name=pod_name, namespace=namespace)
        for container in pod.status.container_statuses:
            container_name = container.name
            container_state = container.last_state

            if container_state.terminated:
                exit_code = container_state.terminated.exit_code
                #print(f"Container '{container_name}' exit code: {exit_code}")

                # Check if exit code is 0 (normal termination)
                if exit_code == 0:
                    # Container command end with normal state
                    return f"Container '{container_name}' exit. But it shoud not exit. Please add 'sleep infinity' command."
                else:
                    return f"Container '{container_name}' failed with exit code {exit_code}."
            else:
                pass
        logs = get_pod_logs(v1, pod_name, namespace)
        error_index = check_log_error(logs)
        if error_index:
            return "Error occurs while configuring VNF. Logs:\n"+ logs [error_index:]
        if time.time() - start_time > wait_time:
            # error or exit did not occur while wait_time
            break
        time.sleep(5)

    # Let's check if 'sleep infinity' working in container.
    try:
        command = [
            "kubectl", "exec", pod_name, "-n", namespace,
            "--", "ps", "aux" 
        ]        
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)        
        if result.returncode == 0:
            processes = result.stdout            
            if not "sleep infinity" in processes:
                return "sleep infinity is NOT running. Please add 'sleep infinity' command."
        else:
            return "Error occur while fetching Pod processes: " + str(result.stderr)

    except Exception as e:
        return "Error occur while fetching Pod processes: " + str(e)

    if vnf == 'firewall':
        input_operation , output_operation, exactly = 'sudo iptables -L -v -n', 'DROP', False
    elif vnf == 'Haproxy':
        input_operation , output_operation, exactly = 'ps -ef | grep haproxy | wc -l', '2', True
    elif vnf == 'nDPI':
        input_operation , output_operation, exactly = 'ps aux', 'ndpiReader', False
    elif vnf == 'ntopng':
        input_operation , output_operation, exactly = 'ps aux', 'ntopng', False
    elif vnf == 'Suricata':
        input_operation , output_operation, exactly = 'ps -ef | grep suricata | wc -l', '2', True 
    else:
        print('weried...')
    result = run_config(v1, pod_name, namespace, input_operation, output_operation, exactly)
    if result == True:
        if vnf == 'Haproxy':
            result = run_config(v1, pod_name, namespace, 'haproxy -c -f /etc/haproxy/haproxy.cfg', 'Configuration file is valid', False)
            if result == True:
                return True
            else:
                return "When I put 'haproxy -c -f /etc/haproxy/haproxy.cfg' operation in Pod to check the VNF, but got this results. \n"+ result+ "It should return 'Configuration file is valid'."
        return True
    else:
        return f"When I put '{input_operation}' operation in Pod to check VNF, it return:\n"+result+ f"It should return '{output_operation}'."

def delete_pod(v1, pod_name, namespace=namespace, logging_=False):
    try:
        response = v1.delete_namespaced_pod(name=pod_name, namespace=namespace)
        if logging_:
            print(f"Pod '{pod_name}' delete complete")
            print("response:", response)
    except ApiException as e:
        if logging_:
            if e.status == 404:
                print(f"Can not find Pod '{pod_name}'")
            else:
                print("Pod deletion error", e)
        else:
            return

def delete_all_pods(v1, apps_v1, namespace=namespace):
    pods_list = list_pods_in_namespace(v1, namespace)
    for pod_name in pods_list:
        delete_pod(v1, pod_name, namespace)
    deployments = apps_v1.list_namespaced_deployment(namespace=namespace)
    for deploy in deployments.items:
            deploy_name = deploy.metadata.name
            apps_v1.delete_namespaced_deployment(
                name=deploy_name,
                namespace=namespace,
                body=client.V1DeleteOptions(propagation_policy='Foreground')
            )


def check_pod_errors(v1, namespace):
    pods = v1.list_namespaced_pod(namespace=namespace)
    for pod in pods.items:
        pod_name = pod.metadata.name
        pod_phase = pod.status.phase
        if pod_phase not in ["Running", "Succeeded"]:
            return False
        if pod.status.container_statuses:
            for cs in pod.status.container_statuses:
                if cs.state.waiting is not None:
                    waiting_reason = cs.state.waiting.reason
                    if waiting_reason in ["CrashLoopBackOff", "ErrImagePull", "ImagePullBackOff", "Error"]:
                        return False
                if cs.state.terminated is not None:
                    terminated_reason = cs.state.terminated.reason
                    if terminated_reason and terminated_reason != "Completed":
                        return False
    return True # None of them are failed.

def restart_daemons(v1):
    try:
        # restart kube-proxy
        pods = v1.list_namespaced_pod(namespace='kube-system', label_selector='k8s-app=kube-proxy')
        for pod in pods.items:
            v1.delete_namespaced_pod(name=pod.metadata.name, namespace='kube-system')
        # restart kube-flannel
        pods = v1.list_namespaced_pod(namespace='kube-flannel')
        for pod in pods.items:
            v1.delete_namespaced_pod(name=pod.metadata.name, namespace='kube-flannel')
    except ApiException as e:
        print(f"Exception when restarting demons : {e}")
    time.sleep(17)