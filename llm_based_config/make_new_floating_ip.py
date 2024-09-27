import openstack
from secret import JUMP_HOST_IP
import os
import base64

def make_new_floating_ip(conn):
    image_name = 'u20.04_x64'
    flavor_name = 'vnf.generic.2.1024.10G'
    network_names = ['NI-data', 'NI-management']
    server_name = 'server-vm'
    desired_floating_ip = JUMP_HOST_IP
    user_data = """
    #cloud-config
    password: dpnm
    chpasswd: { expire: False }
    ssh_pwauth: True
    """
    encoded_user_data = base64.b64encode(user_data.encode('utf-8')).decode('utf-8')  
    image = conn.compute.find_image(image_name)
    flavor = conn.compute.find_flavor(flavor_name)
    networks = [conn.network.find_network(name) for name in network_names]

    #print(f"Creating VM: {server_name}")
    server = conn.compute.create_server(
        name=server_name,
        image_id=image.id,
        flavor_id=flavor.id,
        user_data=encoded_user_data,
        networks=[{"uuid": network.id} for network in networks]
    )

    server = conn.compute.wait_for_server(server)
    #print(f"VM {server_name} is active")

    floating_ip = conn.network.find_ip(desired_floating_ip)
    if not floating_ip:
        print(f"Creating Floating IP: {desired_floating_ip}")
        floating_ip = conn.network.create_ip(floating_network_id='public')
        conn.network.update_ip(floating_ip, floating_ip_address=desired_floating_ip)
        print(f"Created Floating IP: {floating_ip.floating_ip_address}")
        return False
    port = None
    for p in conn.network.ports(device_id=server.id):
        port = p
        break

    if port:
        conn.network.add_ip_to_port(port, floating_ip)
        #print(f"Assigned Floating IP {floating_ip.floating_ip_address} to VM {server.name}")
    else:
        print("No port found for the server.")
        return False

    #print("VM creation and Floating IP assignment complete.")
    os.system('ssh-keygen -R '+desired_floating_ip)
    return server
if __name__ == "__main__":
    conn = openstack.connect(cloud='postech_cloud')
    server = make_new_floating_ip(conn)
    if server:
        print('make new floating IP successfully')
        conn.compute.delete_server(server)
    else:
        print('fail')
