from langchain_community.llms import Ollama
from docx import Document
import os
import re
import openstack
from tqdm import tqdm


if __name__ == '__main__':
    #print(llm.invoke("Tell me a joke"))
    mop_file_path = '../mop/OpenStack_v1/'
    mop_list = os.listdir(mop_file_path)   
    config_file_path = 'OpenStack_Conf/' 
    code_pattern = r'```python(.*?)```'
    code_pattern_second = r'```(.*?)```'

    flavor_name = 'vnf.generic.2.1024.10G'
    image_name = 'u20.04_x64'
    network_name = "'NI-data' and 'NI-management'"
    cloud_name = 'postech_cloud'
    
    prompts = 'You are an OpenStack cloud expert. ' +  \
    'Here is example code to create a VM in OpenStack using python with openstacksdk. '+ \
    "import openstack\nconn = openstack.connect(cloud='openstack_cloud')\n"+ \
    "conn.create_server('test-vm', image = 'ubuntu18.04_image', flavor = 'm1.small', wait = True, network = ['net1', 'net-mgmt'])\n"+ \
    f"OpenStack server and authentication details are in config file. Cloud name is '{cloud_name}'.\n" + \
    'Here is a Method Of Procedure (MOP),'+ \
    'which describes the process of installing a VM in OpenStack and installing specific VNF on the VM. '+ \
    'With reference to this, please write the Python code that automates the process. \n' + \
    'Put the code in the function name "create_vm" and return the server object if the VM is created successfully, ' + \
    "and return False if it fails. Don't put usage or example \n" + \
    f"Use '{image_name}' image, '{flavor_name}' flavor, {network_name} network. \n" 
    
    model_list= ["qwen2", "llama3.1", "llama2"]
    all_mop_num = len(mop_list)
    success_num = {}
    for model in model_list:
        success_num[model] = 0
    vm_num = {}
        
    for mop_file in tqdm(mop_list[:10]):
        vnf = mop_file.split('_')[1]
        if vnf not in vm_num:
            vm_num[vnf] = 1
        else:
            vm_num[vnf] += 1
        
        # Parameters for VM creation
        vm_name = 'vm-'+vnf+'-'+str(vm_num[vnf])
        mop=''
        assert (mop_file.endswith('.docx'))
        doc = Document(mop_file_path + mop_file)
        for para in doc.paragraphs:
            mop += para.text + '\n'
        for model in ["qwen2", "llama3.1", "gemma2", "llama2"]:
            llm = Ollama(model=model)
            llm_response=llm.invoke(prompts+mop)
            #print(llm_response)
            try:
                python_code = re.findall(code_pattern, llm_response, re.DOTALL)
                if not python_code:
                    python_code = re.findall(code_pattern_second, llm_response, re.DOTALL)
            except:
                #print('parsing fail')
                continue
            if not python_code:
                continue
            file_name = f'config_{vnf}_{model}_{vm_num[vnf]}.py'
            with open(config_file_path + file_name, 'w') as f:
                f.write(python_code[0])
            try:
                create_vm = __import__(config_file_path[:-1] + '.' + file_name[:-3],fromlist=['create_vm'])
                server = create_vm.create_vm()
                if server:
                    print(f"VM created successfully with name: {server.name}")
                    success_num[model] += 1
                    conn = openstack.connect(cloud=cloud_name)
                    conn.delete_server(server.name)
            except:
                #print('VM creation failed')
                continue
    print(f"Total MOPs: {all_mop_num}")
    for model in model_list:
        print(f"Model: {model}, Success: {success_num[model]}")