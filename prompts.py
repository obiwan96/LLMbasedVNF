prompts={
    'ansible' :[
    "Please write an Ansible file that installs a container with 2 GB RAM and 2 CPUs on the node named 'kubeworker1' in Kubernetes, assigns the IP '192.168.2.2' to the container, installs a firewall, and configures it to allow only IPs in the 192.168.2.0 subnet to pass through.",
    "Please create an Ansible playbook that deploys a container with 2 GB of RAM and 2 CPUs on the Kubernetes node 'kubeworker1'. The container should be assigned the IP address '192.168.2.2'. Additionally, install a firewall on the node and configure it to only allow traffic from the '192.168.2.0/24' subnet.",
    "Generate an Ansible YAML file that provisions a container on the Kubernetes node 'kubeworker1' with 2 GB RAM and 2 CPUs, assigning it the IP address '192.168.2.2'. Ensure that a firewall is installed and configured to permit traffic exclusively from the '192.168.2.0/24' subnet.",
    "Write an Ansible playbook to deploy a Kubernetes container on the node 'kubeworker1' with 2 GB of memory and 2 CPUs. Assign the container a static IP of '192.168.2.2'. Also, set up a firewall that restricts access to the '192.168.2.0/24' subnet.",
    "Please compose an Ansible playbook for Kubernetes that creates a container on the 'kubeworker1' node with 2 GB of RAM and 2 CPUs. Assign the container the IP address '192.168.2.2', then install and configure a firewall to allow only traffic from the '192.168.2.0/24' subnet.",
    "Create an Ansible YAML script that installs a container on the 'kubeworker1' Kubernetes node with 2 GB of RAM and 2 CPUs, assigns the container the IP '192.168.2.2', and configures a firewall on the node to accept connections only from the '192.168.2.0/24' subnet."
    ],
    'mop' : ["Please write an MOP (Method of Procedure) file about that installing a {container} in"+
             " the node named '{nodename}' in {system}, assigns the IP to that, installing a {function}, and {additional_command}."+
             "Please also write down how to set cpu or ram as user want.",
             
             "Please create an MOP (Method of Procedure) that describe everything that goes through the following process in detail."+
             " Deploys a {container} with specific memory of RAM and specific number of CPUs on the {system} node '{nodename}'."+
             " Set the IP address to specific ip address, additionally, install a {function} on the node and {additional_command}.",
             
             "Write an MOP (Method of Procedure) to deploy a {system} {container} on the node '{nodename}. "+
             "Also, include how to assign the {container} a static IP, install a {function} on that {container} and {additional_command}. Also, include how to set the CPU and RAM as user want.",
    ]
}

system_container_list= [['Kubernetes', 'container'], ['OpenStack', 'VM']]

#container_list = ['container', 'VM']

node_name_list = ['kubeworker1', 'kubeworker2', 'kubeworker3']

function_list = ['firewall', 'Haproxy', 'nDPI', 'ntopng', 'Suricata']

additional_command_list = {'firewall': 
                           ['configures it to allow only IPs in the specific subnet to pass through', 
                            'configures it to block all traffic except for the specific subnet', 
                            'configures it to allow only specific ports to pass through', 
                            'configures it to block all traffic except for specific ports'],
                           'Haproxy':
                           ['Haproxy can be installed with apt. Configures it to load balance between the specific servers',
                            'Haproxy can be installed with apt. Configures it to redirect traffic to the specific server',
                            'Haproxy can be installed with apt. Configures it to cache the specific content',
                            'Haproxy can be installed with apt. Configures it to allow only specific ports to pass through'],
                           'nDPI':
                           ['nDPI can be installed with git. Configures it inspect the packtes of a specific subnet',
                            'nDPI can be installed with git. Configures it to block the specific traffic'],
                           'ntopng':
                           ['ntopng can be installed with apt. Configures it to report the usage by protocol'],
                           'Suricata':
                           ['Suricata can be installed with apt. Please set the most basic rules and set them to work']
}
