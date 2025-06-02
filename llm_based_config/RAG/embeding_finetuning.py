from sentence_transformers import InputExample
from sentence_transformers import SentenceTransformer, losses, models, InputExample, SentencesDataset
from torch.utils.data import DataLoader
import json

model = SentenceTransformer('all-MiniLM-L6-v2')
# example of examples for training
'''train_examples = [
    InputExample(texts=["What is melange?", "Melange is the spice found on Arrakis."]),
    InputExample(texts=["Who is Paul?", "Paul Atreides is a noble born on Caladan."]),
]'''
with open(f"evaluation_data/bad_log_linked.json", 'r') as f:
    log_ground_truth = json.load(f)
train_examples = []
for item in log_ground_truth:
    # Each item has 'log' and 'title' fields
    train_examples.append(InputExample(texts=[item['log'], item['title']]))
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)

train_loss = losses.MultipleNegativesRankingLoss(model)

model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=20,
    warmup_steps=10,
    output_path='./fine_tuned_model/all-MiniLM-L6-v2-finetuned'
)
print ('Model fine-tuned and saved successfully.')
