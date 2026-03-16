prompts={
    'ansible' :[
    "Please write an Ansible file that installs a container with 2 GB RAM and 2 CPUs on the node named 'kubeworker1' in Kubernetes, assigns the IP '192.168.2.2' to the container, installs a firewall, and configures it to allow only IPs in the 192.168.2.0 subnet to pass through.",
    "Please create an Ansible playbook that deploys a container with 2 GB of RAM and 2 CPUs on the Kubernetes node 'kubeworker1'. The container should be assigned the IP address '192.168.2.2'. Additionally, install a firewall on the node and configure it to only allow traffic from the '192.168.2.0/24' subnet.",
    "Generate an Ansible YAML file that provisions a container on the Kubernetes node 'kubeworker1' with 2 GB RAM and 2 CPUs, assigning it the IP address '192.168.2.2'. Ensure that a firewall is installed and configured to permit traffic exclusively from the '192.168.2.0/24' subnet.",
    "Write an Ansible playbook to deploy a Kubernetes container on the node 'kubeworker1' with 2 GB of memory and 2 CPUs. Assign the container a static IP of '192.168.2.2'. Also, set up a firewall that restricts access to the '192.168.2.0/24' subnet.",
    "Please compose an Ansible playbook for Kubernetes that creates a container on the 'kubeworker1' node with 2 GB of RAM and 2 CPUs. Assign the container the IP address '192.168.2.2', then install and configure a firewall to allow only traffic from the '192.168.2.0/24' subnet.",
    "Create an Ansible YAML script that installs a container on the 'kubeworker1' Kubernetes node with 2 GB of RAM and 2 CPUs, assigns the container the IP '192.168.2.2', and configures a firewall on the node to accept connections only from the '192.168.2.0/24' subnet."
    ],
        'mop' : [
             "Write a complete MOP (Method of Procedure) for deploying a {container} on {system}, installing {function}, and executing this required task: {additional_command}. "
             "The MOP must include prerequisites, VM/container resource requirements, assumptions, step-by-step commands, verification checks, rollback steps, and troubleshooting guidance. "
             "Use explicit command examples and expected outputs where relevant.",

             "Create an operationally ready MOP that covers the full lifecycle in detail: preparation, deployment of a {container} on {system}, installation of {function}, and post-install configuration task: {additional_command}. "
             "Include risk/impact notes, validation criteria, and a final acceptance checklist. "
             "The procedure should be executable by an engineer with minimal ambiguity."
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
                           [['subnet','Configure firewall rules to allow only traffic from a specific subnet.'], 
                            ['subnet','Configure firewall rules to block all traffic except a specific subnet.'],  
                            ['port','Configure firewall rules to allow only specific ports to pass through ussing firewall'], 
                            ['port','Configure firewall rules to block all traffic except a specific port.']],
                            'Haproxy':
                           [['loadbalance','Install HAProxy using apt and configure load balancing across specific backend servers.'],
                            ['redirect','Install HAProxy using apt and configure HTTP/HTTPS redirection to a specific backend server.']],
                            #'Haproxy can be installed with apt. Configures it to cache the specific content',
                            #'Haproxy can be installed with apt. Configures it to allow only specific ports to pass through'],
                           'nDPI':
                           [['inspect','Install nDPI from source (git) and configure packet inspection for a specific subnet.'],
                            ['block','Install nDPI from source (git) and configure blocking of specific traffic patterns using nDPI (not ufw).']],
                           'ntopng':
                           [['report','Install ntopng using apt and configure protocol-based traffic usage reporting.']],
                           'Suricata':
                           [['basic','Install Suricata using apt, apply basic detection rules, and verify that alerting works properly.']]
}
