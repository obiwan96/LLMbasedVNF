from openai import OpenAI
from secret import OPENAI_API_KEY, KUBER_PASSWORD
from prompts import prompts
import re

openai_client = OpenAI(api_key=OPENAI_API_KEY)
success_count = 1
data_path='data/'

for prompt in prompts:
    response = openai_client.chat.completions.create(
        model = 'gpt-4o-mini', messages=[
            {"role" : "system", "content" : 'You are an expert in Kubernetes management.'},
            {"role" : "user", "content" : prompt}
            ], temperature=0)
    content=response.choices[0].message.content
    #print(content)
    #print('-------------------')
    yaml_pattern = r"```yaml(.*?)```"
    yaml_match = re.search(yaml_pattern, content, re.DOTALL)
    if yaml_match:
        yaml_content = yaml_match.group(1).strip()
        print("Extracted YAML content:")
        #print(yaml_content)

        yaml_file_path = data_path+"kubernetes_setup_"+str(success_count)+".yaml"
        with open(yaml_file_path, "w") as yaml_file:
            yaml_file.write(yaml_content)
        
        print(f"YAML content has been saved to {yaml_file_path}")
        
        # Ansible Playbook 실행
        import subprocess
        
        ansible_command = ["ansible-playbook", yaml_file_path, '--extra-vars', '"ansible_sudo_pass='+KUBER_PASSWORD+'"']
        
        try:
            result = subprocess.run(ansible_command, capture_output=True, text=True)
            if result.returncode == 0:
                print("Ansible Playbook executed successfully.")
                success_count += 1
            else:
                print("Ansible Playbook execution failed.")
                print("Error details:", result.stderr)
                
            # 출력 결과
            print("Output:", result.stdout)
        except Exception as e:
            print("Failed to execute Ansible Playbook.")
            print(str(e))
    else:
        print("YAML content not found.")

print(f'Total {success_count-1} YAML files have been created and executed successfully.')