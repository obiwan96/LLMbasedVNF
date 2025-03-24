import ansible_runner
from kubernetes import client, config, stream
from kubernetes.client.rest import ApiException
import yaml
import re
from prompt import namespace
import time
import os

def list_pods_in_namespace(v1, namespace=namespace):
    pods = v1.list_namespaced_pod(namespace)
    
    return [pod.metadata.name for pod in pods.items]

def update_yml(data):
    if isinstance(data, dict):
        for key, value in data.items():
            if key == 'namespace':
                data[key] = '{{ namespace }}'
            if key == 'definition' and 'metadata' in value:
                data[key]['metadata']['name'] = '{{ pod_name }}'
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

def check_yml_if_do_in_localhost(data):
    for play in data:
        if "hosts" in play:
            if str(play["hosts"]).strip().lower() == "localhost":
                if "tasks" in play:
                    for task in play["tasks"]:
                        if not 'k8s' in str(task) and not 'kubernetes' in str(task):
                            # Suspect. It may do something in localhost.
                            return False
    return True

class OutputCollector:
    def __init__(self):
        self.captured_output = ''

    def event_handler(self, event_data):
        if 'stdout' in event_data and event_data['stdout']:
            self.captured_output+=event_data['stdout']

def test_creation_ansible(llm_response, vnf, model, vm_num, v1, timeout=300):
    config_file_path = 'K8S_Conf/'
    if not os.path.exists(config_file_path):
        os.makedirs(config_file_path)
    code_pattern = r'```yaml(.*?)```'
    code_pattern_second = r'```(.*?)```'

    try:
        yml_code = re.findall(code_pattern, llm_response, re.DOTALL)
        if not yml_code:
            yml_code = re.findall(code_pattern_second, llm_response, re.DOTALL)
    except:
        #print('parsing fail')
        return False, "I can't see YAML code in your response."
    if not yml_code:
        return False, "I can't see YAML code in your response."
    try:
        pod_name= vnf.lower()+'-pod'
        yaml_data = yaml.safe_load(yml_code[0])
        update_yml(yaml_data)
        if not check_yml_if_do_in_localhost(yaml_data):
            return False, f"Your YAML code do something in localhost, but you don't anything to do in localhost. Please modify it to work in Kubernetes."
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
            extravars = {'pod_name': pod_name, 'namespace' : namespace, 'image': 'dokken/ubuntu-20.04', 'image_name': 'dokken/ubuntu-20.04'},
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
                    if time.time() - start_time > timeout:
                        return False, f"Ansible run succeed, but the containers are not ready whitin {timeout} seconds. Test fail."
                    time.sleep(5)
            except ApiException as e:
                if e.status == 404:
                    return False, f"Ansible run succeeed. But Pod '{pod_name}' doesn't exist in the Kubernetes namespace '{namespace}."
                else:
                    return False, "Error while searching Pod"
            
        else:
            return False, 'Ansible run failed. Ansible runner status: '+runner.status + '\n Here are Ansible running results:\n'+collector.captured_output

    except Exception as e:
        #print(e)
        #print('VM creation failed')
        return False, e

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
        return e
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

def test_K8S_configuration(pod_name, vnf, model, vm_num, v1, namespace):
    config_file_path = 'K8S_Conf/'
    if vnf == 'firewall':
        result = run_config(v1, pod_name, namespace, 'sudo iptables -L -v -n', 'DROP')
    elif vnf == 'Haproxy':
        result = run_config(v1, pod_name, namespace, 'systemctl is-active haproxy', 'active', exactly=True)
        if result == True:
            result = run_config(v1, pod_name, namespace, 'haproxy -c -f /etc/haproxy/haproxy.cfg', 'Configuration file is valid')
    elif vnf == 'nDPI':
        result = run_config(v1, pod_name, namespace, 'ps aux', 'ndpiReader')
    elif vnf == 'ntopng':
        result = run_config(v1, pod_name, namespace, 'ps aux', 'ntopng')
    elif vnf == 'Suricata':
        result = run_config(v1, pod_name, namespace, 'systemctl is-active suricata', 'active', exactly=True)
    else:
        print('weried...')
    if result == True:
        return True
    elif result == False:
        return 'Your code is run well, but when I check the VM, VNF is not installed correctly as intended. When I run it, it return:\n'+result
    else:
        return result        

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