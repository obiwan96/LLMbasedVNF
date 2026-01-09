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
    print (f"Good example code for {lang} in {system} is readed. {len(good_example)} files found.")
    good_example_prefix=''
    good_example_suffix=''
    '''for example in good_example:
        good_example_str += example+':\n'+good_example[example] +'\n' '''
    good_example_str ="\nPlease rememeber, these are example code, so you have to just refer to them. For detailed VNF setup methods and parameters, follow the upper description, not the example code."
    if lang == 'Python':
        if system == 'OpenStack':
            good_example_prefix = "Here are 'example' codes to create and configurate a VNF in OpenStack using python with openstacksdk. \n"
            prompts_1 = '''You are an OpenStack cloud expert. '''+ \
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

        # Kubernetes Python code part is just copy version of OpenStack. need change.
        elif system == 'Kubernetes':
            good_example_prefix = "\nMOP ends.\nAlso, here are 'example' codes to create and configure a VNF in Kubernetes using Python with Kubernetes library for reference. \n" 
            prompts_1 = f'''You are an Kubernetes cloud expert. 
            Please write Python code that creates a Kubernetes Pod and installs a VNF inside it, taking the following points into consideration.
            You don't have to explain your code. Keep your answer code-focused and as simple as possible.
            I will provide you with a Method Of Procedure (MOP) that describes the process of installing a Pod in Kubernetes and configuring a VNF on that Pod. 
            Based on this MOP, you must write a single, self-contained Python function with the following strict requirements:
	        1.	The entire logic for creating the Pod and configuring the VNF must be contained within one function only; do not split the logic into multiple functions or files.
	        2.	The function must be named 'create_pod'.
	        3.	The 'create_pod' function must accept exactly three input parameters: 'pod_name', 'namespace', and 'image_name'.
	        4.	The function must return True if the Pod is created and the VNF is configured successfully; otherwise, it must return False.
	        5.	Do not include any additional parameters or helper functions; everything must be implemented inside the create_pod function.
            6.  Don't put any kind of infinity loop in the code.
            7.  The Kubernetes configuration file is in it's default path, '/home/dpnm/.kube/config'. So to load the configuration, use 'load_kube_config' instead of 'load_incluster_config'.
            8.  Don't put usage or example in the code block.
            9.  Instead of using the cluster's default DNS settings, manually set the DNS to '{DNS_IP}'.
            10. Put 'sleep infinity' command, so that container dosen't killed.
            12. Since systemctl cannot be used in containers, even if the MOP instructs to install the VNF using systemctl, an alternative method like running with daemon option, must be found.
            13. Don't use stdin to get any kind of password.
            14. Don't make 'host_name' option as True in Pod creation step.
            15. Configure the VNF in one container.

            Please ensure that you follow these instructions exactly and do not deviate from the specified function name, parameter list, or return value.'''
        
            prompts_2= f'''Here is the MOP: '''

    elif lang == 'Ansible':
        if system== 'OpenStack':
            return 'need to implement'
        # Let's do Kubernetes Ansible first!
        elif system== 'Kubernetes':
            good_example_prefix = "\nMOP ends. Here are example YAML configuration snippets for creating and configuring a CNF in Kubernetes using Ansible. \n"
            good_example_suffix = "\nNow, write the YAML code that automates the process described in the MOP, following the above requirements."
            prompts_1 = '''You are a Kubernetes cloud expert. 
            I will provide you with a Method of Procedure (MOP), which outlines the steps for deploying a Pod in Kubernetes and installing and configuring the CNF specified in the MOP on that Pod.
            Based on this, please write the corresponding Ansible YAML code that automates the installation and configuration process described in the MOP with the following strict requirements:\n'''
            
            prompts_2= f'''1. Kubernetes is already properly installed on a remote server, and the kubeconfig file needed to connect to it is located at the default path. I can access the Kubernetes server from my current (local) server.
            2. Based on this setup, write a YAML code that fully automate the process of creating a Pod and installing a VNF inside it.
            3. Do not write any code that performs direct actions on the local server or inside the Kubernetes Node.
            4. You must keep only the following three values as variables: pod_name, namespace, and image_name, so that they can be passed in separately.
            5. You should keep the pod name, namespace, and image fields as variables, so they can be passed separately. The variable names are 'pod_name', 'namespace', and 'image_name'. Aside from these three, do not use any other variables — all other values should be hardcoded explicitly.
            6. Do not include a vars section in the code, as I will run it using my own variables.
            7. Instead of using the cluster’s default DNS settings, manually set the DNS to {DNS_IP}.
            8. Include the command sleep infinity in the container so that it stays alive and doesn’t get terminated.
            9. Do not set the host_name option to True. 
            10. Keep in mind that systemctl cannot be used inside containers. Even if the MOP instructs installing the VNF using systemctl, you must find an alternative method, such as running it with a daemon option.
            
            The final goal is that I should be able to create the desired Pod simply by providing the values for pod_name, namespace, and image_name, and executing the Ansible playbook you provide without any modifications to the code.
            Write the entire code as a single code block, and do not split it into multiple pieces.
            
            Here is the MOP: '''

    return prompts_1, prompts_2, (good_example, good_example_prefix, good_example_suffix)