from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead

import torch

# Load model and tokenizer
model_name = "gpt2"
# 1) 토크나이저에 pad_token 추가
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})

# 2) TRL 래퍼 모델 로드
model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name)

# 3) 래퍼가 감싸고 있는 HF 모델에 resize 호출
#model.pretrained_model.resize_token_embeddings(len(tokenizer))
model_ref = AutoModelForCausalLMWithValueHead.from_pretrained(model_name)

# PPO Configuration
ppo_config = PPOConfig(
    model_name=model_name,
    batch_size=1,
    forward_batch_size=1,
    log_with=None  # or "wandb"
)

# Dataset: prompts for LLM to generate from
prompts = ["What is the capital of France?", "Tell me a joke."]

# Tokenize prompts
#tokenizer.pad_token = tokenizer.eos_token
#model.resize_token_embeddings(len(tokenizer))

tokenized_prompts = tokenizer(prompts, return_tensors="pt", padding=True).input_ids

def compute_custom_reward(prompt, response):
    """
    Custom reward function that computes a reward based on the response.
    This is a placeholder function and should be replaced with actual logic.
    """
    # Example logic: reward is higher for responses containing "Paris"
    if "Paris" in response:
        return 1.0  # Positive reward
    else:
        return -1.0  # Negative reward

# Custom reward function
def custom_reward_fn(prompts, responses):
    rewards = []
    for prompt, response in zip(prompts, responses):
        rewards.append(compute_custom_reward(prompt, response))
    return rewards

# Trainer
ppo_trainer = PPOTrainer(
    config=ppo_config,
    model=model,
    ref_model=model_ref,
    tokenizer=tokenizer
)

# Main training loop
for step, prompt_text in enumerate(prompts, start=1):
    # Generate model response
    input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids
    query_tensors = [input_ids[i] for i in range(input_ids.size(0))]
    response_ids = ppo_trainer.generate(query_tensors, max_new_tokens=20)
    response_text = tokenizer.decode(response_ids[0], skip_special_tokens=True)

    # Compute reward
    reward = compute_custom_reward(prompt_text, response_text)

    # Run PPO step
    stats = ppo_trainer.step(
        query_tensors,          # ← List[Tensor(seq_len,)]
        [response_ids[0]],       # ← List[Tensor(new_seq_len,)]
        [torch.tensor(reward)]                 # ← List[float]
    )
    print(f"Step {step}: Response: {response_text}, Reward: {reward}, Stats: {stats}")
print('done')