import torch
from transformers import AutoTokenizer, BitsAndBytesConfig
from transformers import AutoModelForCausalLM
from transformers import AutoModelForSequenceClassification

from peft import get_peft_model, LoraConfig, TaskType
from trl import PPOTrainer, PPOConfig
from datasets import Dataset

model_name = "google/gemma-3-27b-it"
bnb_config = BitsAndBytesConfig(load_in_8bit=True, llm_int8_threshold=6.0)

tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    quantization_config=bnb_config,
    torch_dtype=torch.float16
)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],  # Need change?
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)
model = get_peft_model(base_model, lora_config)
model_ref = get_peft_model(base_model, lora_config)

#PPO + LoRA
ppo_config = PPOConfig(
    learning_rate=1e-5,
    batch_size=2,
    mini_batch_size=1,
    remove_unused_columns=False
)

dataset = Dataset.from_dict({
    "prompt": [
        "Explain black holes in simple terms.",
        "What is quantum entanglement?",
        "Tell me a motivational quote."
    ]
})
reward_model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased") # 내부에서 reward 직접 주기 때문에 dummy reward model 둠.

ppo_trainer = PPOTrainer(ppo_config,tokenizer,model,model_ref,reward_model=reward_model,train_dataset=dataset) 

for batch in dataset:
    prompt = batch["prompt"]
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    response_ids = model.generate(
        input_ids=inputs["input_ids"],
        max_new_tokens=64,
        do_sample=True,
        top_k=50,
        top_p=0.95
    )
    
    response_text = tokenizer.decode(response_ids[0], skip_special_tokens=True)
    reward = 1.0 if "inspire" in response_text.lower() or "quantum" in response_text.lower() else 0.0

    print(f"Prompt: {prompt}")
    print(f"Response: {response_text}")
    print(f"Reward: {reward}\n")

    ppo_trainer.step([inputs["input_ids"][0]], [response_ids[0]], [torch.tensor(reward)])

model.save_pretrained("./llama3-lora-ppo")
