import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DistilBertConfig, DistilBertForSequenceClassification, DataCollatorWithPadding
from datasets import load_dataset

dataset = load_dataset("imdb")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"using {device}.")

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

tokenized_dataset = dataset.map(lambda example: tokenizer(example['text'], truncation=True), batched=True, batch_size=32)

tokenized_dataset = tokenized_dataset.remove_columns("text")
tokenized_dataset = tokenized_dataset.rename_column("label", 'labels')
tokenized_dataset.set_format("torch")

collate_fn = DataCollatorWithPadding(tokenizer=tokenizer)

tokenized_dataset["train"] = tokenized_dataset["train"].train_test_split(shuffle=True, test_size=0.2)

train_dataloader = DataLoader(tokenized_dataset['train']["train"], batch_size=16, shuffle=True, collate_fn=collate_fn)
val_dataloader = DataLoader(tokenized_dataset["train"]['test'], batch_size=16, shuffle=False, collate_fn=collate_fn)
test_dataloader = DataLoader(tokenized_dataset['test'], batch_size=16, shuffle=False, collate_fn=collate_fn)

step_list = [2000, 5000, 20000, 50000, 100000, 'best']
num_epochs = 3

for pretrain_step in step_list:
  best_val_loss = float("inf")
  if pretrain_step == 0:
    config = DistilBertConfig(num_labels=2)
    model = DistilBertForSequenceClassification(config).to(device)
  else:
    model = DistilBertForSequenceClassification.from_pretrained(f"./checkpoints/DistilBERT_wikitext103v1_{pretrain_step}.pt", num_labels=2).to(device)

  optimizer = optim.AdamW(model.parameters(), lr=3e-5)
  for epoch in range(num_epochs):
    for batch in train_dataloader:
      model.train()
      batch = batch.to(device)
      optimizer.zero_grad()
      outputs = model(**batch)
      outputs.loss.backward()
      optimizer.step()

    model.eval()
    val_loss = 0
    for batch in val_dataloader:
      batch = batch.to(device)
      with torch.no_grad():
        outputs = model(**batch)
        val_loss += outputs.loss * len(batch)

    print(f"Dev loss: {val_loss / len(tokenized_dataset['train']['test'])}")
    if val_loss / len(tokenized_dataset['train']['test']) < best_val_loss:
      print("Saving checkpoint!")
      best_val_loss = val_loss / len(tokenized_dataset['train']['test'])
      model.save_pretrained(f"./checkpoints/DistilBERT_wikitext103v1_imdb_{pretrain_step}_best.pt")

  model = DistilBertForSequenceClassification.from_pretrained(f"./checkpoints/DistilBERT_wikitext103v1_imdb_{pretrain_step}_best.pt").to(device)
  model.eval()
  test_loss = 0
  test_correct = 0
  for batch in test_dataloader:
    batch = batch.to(device)
    with torch.no_grad():
      outputs = model(**batch)
      test_loss += outputs.loss * len(batch)
      y_pred = torch.argmax(outputs.logits, dim=-1)
      for i, value in enumerate(y_pred):
        if value == batch["labels"][i]:
          test_correct += 1

  print(f"Test loss: {test_loss / len(tokenized_dataset['test'])}")
  print(f"Accuracy: {test_correct / len(tokenized_dataset['test'])}")