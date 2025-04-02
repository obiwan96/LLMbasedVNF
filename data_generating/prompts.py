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
             " {system}, installing a {function}. {additional_command}."+
             "Please also include the system requirements for VM to install that function. Please write as detailed as possible.",
             
             "Please create an MOP (Method of Procedure) that describe everything that goes through the following process in detail."+
             " Deploys a {container} with system requirements for the {function} in {system}. "+
             " Install a {function} on the {container}. {additional_command} in the {container}. Please write with maximum detail."
             
            # "Write an MOP (Method of Procedure) to deploy a {system} {container} on the specific node. "+
             #"Also, include how to assign the {container} a static IP, install a {function} on that {container} and {additional_command}. Also, include how to set the CPU and RAM as user want.",
    ]
}

#system_container_list= [['Kubernetes', 'container'], ['OpenStack', 'VM']]
system_container_list= [['OpenStack', 'VM']]

#container_list = ['container', 'VM']

#node_name_list = ['kubeworker1', 'kubeworker2', 'kubeworker3']

function_list = ['firewall', 'Haproxy', 'nDPI', 'ntopng', 'Suricata']

additional_command_list = {'firewall': 
                           [['subnet','Configure it to allow only IPs in the specific subnet to pass through ussing firewall'], 
                            ['subnet','Configure it to block all traffic except for the specific subnet ussing firewall'], 
                            #'configures it to allow only specific ports to pass through ussing firewall', 
                            ['port','Configure it to block all traffic except for the specific port ussing firewall']],
                           'Haproxy':
                           [['loadbalance','Haproxy can be installed with apt. Configure it to load balance between the specific servers'],
                            ['redirect','Haproxy can be installed with apt. Configure it to redirect traffic to the specific server']],
                            #'Haproxy can be installed with apt. Configures it to cache the specific content',
                            #'Haproxy can be installed with apt. Configures it to allow only specific ports to pass through'],
                           'nDPI':
                           [['inspect','nDPI can be installed with git. Configure it inspect the packtes of a specific subnet using nDPI'],
                            ['block','nDPI can be installed with git. Configure it to block the specific traffic using nDPI, not ufw']],
                           'ntopng':
                           [['report','ntopng can be installed with apt. Configure it to report the usage by protocol using ntopng']],
                           'Suricata':
                           [['basic','Suricata can be installed with apt. Please set the most basic rules and set them to work']]
}
