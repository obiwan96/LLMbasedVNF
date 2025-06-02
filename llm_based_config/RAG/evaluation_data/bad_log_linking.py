import json 
import os
from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer("all-MiniLM-L6-v2")

def check_sim(text1, text2, threshold=0.85):
    emb1 = model.encode(text1, convert_to_tensor=True)
    emb2 = model.encode(text2, convert_to_tensor=True)
    sim = util.cos_sim(emb1, emb2)
    return sim.item() > threshold

with open('bad_log.txt', 'r')  as f:
    bad_log = f.read()
if 'bad_log_linked.json' in os.listdir('.'):
    with open('bad_log_linked.json', 'r') as f:
        bad_log_data = json.load(f)
else:
    bad_log_data=[]
skipped_logs = []
for line in bad_log.split('\n'):
    if line.strip():  # Check if the line is not empty
        if 'ERR:' in line and 'http://' in line:
            bad_log_data.append({'log': line.strip(), 'title': 'apt-get update fails to fetch files, “Temporary failure resolving …” error'})
            continue
        elif 'Cannot initiate' in line:
            bad_log_data.append({'log': line.strip(), 'title': '"Cannot initiate the connection to in.archive.ubuntu.com:80" when trying to run "sudo apt-get update" behind proxy'})
            continue
        if line.strip() in [item['log'] for item in bad_log_data]:
            print('\033[94mskipped\033[0m')
            continue
        skip_find=False
        for skipped_log in skipped_logs:
            if check_sim(line, skipped_log):
                print('\033[92mskipped based on similarity\033[0m')
                skip_find=True
                break
        if skip_find:
            continue
        for bad_log in bad_log_data:
            if check_sim(line, bad_log['log']):
                print('\033[92mskipped based on similarity\033[0m')
                skip_find=True
                break
        if skip_find:
            continue
        print('\033[94mHere is the logs\033[0m')
        print(line)
        title= input("\033[91mTitle of docs to link (type 'skip' to skip):\033[0m")
        if title.lower()== 'skip':
            print('\033[94mskipped\033[0m')
            skipped_logs.append(line)
            continue
        bad_log_data.append({'log': line.strip(), 'title': title.strip()})
    with open('bad_log_linked.json' , 'w') as f:
        json.dump(bad_log_data, f, indent=4)