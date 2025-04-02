import os
import sys
from secret import JUMP_HOST_IP, JUMP_HOST_PWD

pod_name = 'cnf-pod'
namespace = 'llm-config'
DNS_IP = '10.99.30.112'
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
    good_example_str=''
    for example in good_example:
        good_example_str += example+':\n'+good_example[example] +'\n'

    if lang == 'Python':
        if system == 'OpenStack':
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

        # Kubernetes Python code part is just copy version of OpenStack. need change.
        elif system == 'Kubernetes':
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

    elif lang == 'Ansible':
        if system== 'OpenStack':
            return 'need to implement'
        # Let's do Kubernetes Ansible first!
        elif system== 'Kubernetes':
            prompts_1 = '''You are a Kubernetes cloud expert. 
            I will provide you with a Method of Procedure (MOP), which outlines the steps for deploying a Pod in Kubernetes and installing and configuring the CNF specified in the MOP on that Pod.
            Based on this, please write the corresponding Ansible YAML code that automates the installation and configuration process described in the MOP.

            Below are example YAML configuration snippets for creating and configuring a CNF in Kubernetes using Ansible.\n'''
            good_example_str + \
            f'''\nPlease rememeber, these are example code, so you have to just refer to them.
            For detailed CNF setup methods and parameters, follow the description in the MOP, not the example YAML code.'''
            
            prompts_2= f'''Kubernetes is already properly installed on another server, and the configuration file to connect to it is located at the default path.
            So I can access the Kubernetes server from the current server. 
            Please write code that connects to the Kubernetes server and creates a pod there. 
            Do not write any code that performs direct actions on the local server or inside the Kubernetes pod.
            
            You should keep the pod name, namespace, and image fields as variables, so they can be passed separately.
            The variable names are 'pod_name', 'namespace', and 'image_name'.
            Aside from these three, do not use any other variables—please write all other values explicitly.
            I will run the code using my own variables, so do not include a 'vars' section in the code.
            Also, instead of using the cluster's default DNS settings, manually set the DNS to '{DNS_IP}'.
            For these parts, it would be helpful to refer to the example code.
            You should put 'sleep infinity' command, so that container dosen't killed.
            Rememeber that, since systemctl cannot be used in containers, even if the MOP instructs to install the VNF using systemctl, an alternative method like running with daemon option, must be found.
            Through this, I want to be able to create the desired Pod by simply providing the values for the three variables — 'pod_name', 'namespace', and 'image_name' — and running the Ansible playbook you provide without making any modifications to the code. 
            My goal is to fully automate the MOP process.
            Please write it as a single code block, not separated into multiple pieces.
            Here is the MOP: '''

    return prompts_1, prompts_2