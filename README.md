VNF Deployment Automation using LLM
===
Requirements (python3.7)
```
Ollama, langchain, docx, openstacksdk
```

This is a part of DPNM-SNIC project for VNF Depoyment Automation.

We are conducting an experiment to put MOPs on VNF deployment configuration in various LLM models and create a code/ansible file that automates them.

     - data_generating
        : making MOPs using GPT. K8S and OpenStack are both available with Kor, Eng.
     - llm_based_config
        : LLM based VNF deployment automation experiment framework.
     - mop
        : example MOPs. 
