import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import DistilBertConfig, DistilBertForMaskedLM, AutoTokenizer, DataCollatorForLanguageModeling
from datasets import load_dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"using {device}.")

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased")

config = DistilBertConfig()
model = DistilBertForMaskedLM(config).to(device)

dataset = load_dataset("wikitext", 'wikitext-103-v1')

tokenized_dataset = dataset.map(lambda example: tokenizer(example['text'], truncation=True), batched=True, batch_size=32)

tokenized_dataset = tokenized_dataset.remove_columns("text")
tokenized_dataset.set_format("torch")

collate_fn = DataCollatorForLanguageModeling(tokenizer=tokenizer)

train_dataloader = DataLoader(tokenized_dataset['train'], batch_size=16, shuffle=True, collate_fn=collate_fn)
val_dataloader = DataLoader(tokenized_dataset['validation'], batch_size=16, shuffle=False, collate_fn=collate_fn)

optimizer = optim.AdamW(model.parameters(), lr=1e-4)
num_epochs = 1
best_val_loss = float("inf")
step_list = [10, 100, 500, 2000, 5000, 20000, 50000, 100000]

for epoch in range(num_epochs):
  step = 0
  for batch in train_dataloader:
    model.train()
    batch = batch.to(device)
    optimizer.zero_grad()
    outputs = model(**batch)
    outputs.loss.backward()
    optimizer.step()
    step += 1
    if step in step_list:
        model.save_pretrained(f"./checkpoints/DistilBERT_wikitext103v1_{step}.pt")
    if step % 200 == 0:
      print(f"step {step}...")
      print(f"batch training loss: {outputs.loss}")
    if step % 2000 == 0:
      print("Validating...")
      model.eval()
      val_loss = 0
      for batch_val in val_dataloader:
        batch_val = batch_val.to(device)
        with torch.no_grad():
          outputs_val = model(**batch_val)
          val_loss += outputs_val.loss * len(batch_val)

      print(f"Dev loss: {val_loss / len(tokenized_dataset['validation'])}")
      if val_loss / len(tokenized_dataset['validation']) < best_val_loss:
        print("Saving checkpoint!")
        best_val_loss = val_loss / len(tokenized_dataset['validation'])
        model.save_pretrained("./checkpoints/DistilBERT_wikitext103v1_best.pt")
