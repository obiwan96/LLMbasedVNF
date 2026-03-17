import os
import sys
from secret import JUMP_HOST_IP, JUMP_HOST_PWD

#pod_name = 'cnf-pod'
namespace = 'llm-config'
DNS_IP = '10.244.0.2'

def read_good_example(method, platform, example_path = 'Good_Example/'):
    good_example = {}
    for file in os.listdir(example_path):
        if platform.lower() in file.lower() and file.endswith(method):
            with open(example_path+file, 'r') as f:
                good_example[file] = f.read()
    return good_example

def prompt(lang, system):
    if system=='OpenStack':
        flavor_name = 'vnf.generic.2.1024.10G'
        image_name = 'u20.04_x64'
        network_name = "'NI-data' and 'NI-management'"
        cloud_name = 'postech_cloud'
    elif system == 'Kubernetes':
        flavor_name='ubuntu:18.04'
    if lang == 'Python':
        if system == 'OpenStack':
            good_example= read_good_example('py', 'OpenStack')
        elif system == 'Kubernetes':
            good_example= read_good_example('py', 'k8s')
        else:
            good_example = read_good_example('py')
    elif lang == 'Ansible':
        if system == 'OpenStack':
            good_example= read_good_example('yml', 'OpenStack')
        elif system == 'Kubernetes':
            good_example= read_good_example('yml', 'k8s')
        else:
            good_example = read_good_example('yml')
    else:
        print('Currently, only supporting Python and Ansible. Specify which one you use.')
        sys.exit(0)

    print(f"Good example code for {lang} in {system} is loaded. {len(good_example)} files found.")
    good_example_prefix=''
    good_example_suffix=''

    if lang == 'Python':
        if system == 'OpenStack':
            good_example_prefix = (
                "Here are reference examples for creating and configuring a VNF in OpenStack "
                "using Python and openstacksdk.\n"
                "Use them only as style references. If they conflict with the requirements or the MOP, "
                "follow the requirements and the MOP.\n"
            )
            prompts_1 = f'''You are an OpenStack cloud automation expert.

Generate Python code that automates the full procedure described in the MOP below.

Output requirements:
- Output Python code only.
- Do not include explanations, markdown fences, sample usage, or extra text.
- Put everything in one code snippet.

Required functions:
- Define `create_vm()`.
- `create_vm()` must create the VM and return the server object on success, or `False` on failure.
- Define `config_vm(server)`.
- `config_vm(server)` must configure the VNF on the VM and return `True` on success, or `False` on failure.

Implementation requirements:
- Use OpenStack server and authentication details from the config file.
- Use cloud name `{cloud_name}`.
- Use image `{image_name}`, flavor `{flavor_name}`, and networks {network_name}.
- Do not create or manage a key pair.
- Do not request any value from stdin or input().
- If in-VM access is needed, do not assign a floating IP.
- Access the VM through the Jump Host and SSH into the VM by using the VM IP from the returned server object.
- The VM must support SSH password login so the Jump Host can connect to it.
- Use username `ubuntu` for the Jump Host.
- Jump Host IP: {JUMP_HOST_IP}
- Jump Host Password: {JUMP_HOST_PWD}
- Do not use interactive editors such as vim or nano in Paramiko sessions.
- Use non-interactive shell commands and absolute paths when modifying files inside the VM.
- If the MOP conflicts with the rules above, follow the rules above.
'''

            prompts_2= '''Here is the MOP:
'''

        # Kubernetes Python code part is just copy version of OpenStack. need change.
        elif system == 'Kubernetes':
            good_example_prefix = (
                "\nMOP ends.\n"
                "Also, here are reference examples for creating and configuring a VNF in Kubernetes "
                "using Python and the Kubernetes client library.\n"
                "Use them only as references. If they conflict with the requirements or the MOP, "
                "follow the requirements and the MOP.\n"
            )
            prompts_1 = f'''You are a Kubernetes cloud automation expert.

Generate Python code that creates one Pod and configures the VNF inside that Pod according to the MOP below.

Output requirements:
- Output Python code only.
- Do not include explanations, markdown fences, sample usage, or extra text.
- Define exactly one function: `create_pod(pod_name, namespace, image_name)`.
- Do not define helper functions or additional top-level functions.
- `create_pod(pod_name, namespace, image_name)` must return `True` only when both Pod creation and VNF configuration succeed; otherwise it must return `False`.

Implementation requirements:
- Use the Kubernetes Python client.
- Load Kubernetes configuration from the default kubeconfig path by using `load_kube_config`, not `load_incluster_config`.
- Implement the entire workflow inside `create_pod`.
- Create exactly one Pod with exactly one container.
- Do not use stdin or input() for any password or prompt.
- Do not use any infinite loop in the Python code.
- It is allowed to keep the container alive by setting the container command or args to `sleep infinity`.
- Manually set Pod DNS instead of using the cluster default:
  - `dns_policy` must be `None`
  - `dns_config.nameservers` must contain `{DNS_IP}`
- Do not set `hostname`.
- Since `systemctl` cannot be used inside containers, if the MOP mentions `systemctl`, replace it with a container-safe method such as running the process directly or using a daemon option.
- Wait until the Pod is running before executing VNF configuration commands.
- Configure the VNF only inside that single container.
- Use absolute paths when modifying files in the container.
- Wrap error-prone operations with exception handling and return `False` on failure.
- If the MOP conflicts with the rules above, follow the rules above.
'''

            prompts_2= '''Here is the MOP:
'''

    elif lang == 'Ansible':
        if system== 'OpenStack':
            return 'need to implement'
        # Let's do Kubernetes Ansible first!
        elif system== 'Kubernetes':
            good_example_prefix = (
                "\nMOP ends. Here are reference YAML snippets for creating and configuring a CNF "
                "in Kubernetes using Ansible.\n"
                "Use them only as references. If they conflict with the requirements or the MOP, "
                "follow the requirements and the MOP.\n"
            )
            good_example_suffix = "\nNow, write the YAML code that automates the process described in the MOP, following the above requirements."
            prompts_1 = '''You are a Kubernetes and Ansible automation expert.

Generate Ansible YAML that creates one Pod and configures the VNF inside that Pod according to the MOP below.

Output requirements:
- Output YAML only.
- Do not include explanations, markdown fences, or extra text.
- Write the entire playbook in one snippet.

Implementation requirements:
'''

            prompts_2= f'''1. Kubernetes is already installed on a remote server, and the kubeconfig needed to connect to it is available at the default path.
2. I can reach that Kubernetes server from my current server.
3. Write YAML that fully automates both Pod creation and VNF configuration inside the Pod.
4. Do not perform direct configuration on the local server or on the Kubernetes node filesystem.
5. Use only these three variables: `pod_name`, `namespace`, and `image_name`.
6. Do not declare a `vars` section in the playbook.
7. Do not introduce any other variables; all other values must be hardcoded.
8. Create exactly one Pod with exactly one container.
9. Manually set Pod DNS instead of using the cluster default:
   - `dnsPolicy: None`
   - `dnsConfig.nameservers: [{DNS_IP}]`
10. Keep the container alive with `sleep infinity`.
11. Do not set `hostname`.
12. Since `systemctl` cannot be used inside containers, if the MOP mentions `systemctl`, replace it with a container-safe method such as running the process directly or using a daemon option.
13. The playbook should be executable as-is by passing only `pod_name`, `namespace`, and `image_name`.
14. If command execution inside the Pod is required, use a Kubernetes-aware Ansible method for in-Pod execution rather than node-level access.
15. If the MOP conflicts with the rules above, follow the rules above.

Here is the MOP:
'''

    return prompts_1, prompts_2, (good_example, good_example_prefix, good_example_suffix)
