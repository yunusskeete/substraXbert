# Customising Substra

## Identifying Similarities
The similaritites between the training of pytorch MNIST and pytorch BERT are defined as follows:

### 1. The use of the `torch.utils.data.Dataset` class:
* Both of these are user-defined prior to execution.

#### MNIST:
```python
class TorchDataset(torch.utils.data.Dataset):
    def __init__(self, datasamples, is_inference: bool):
        self.x = datasamples["images"]
        self.y = datasamples["labels"]
        self.is_inference = is_inference

    def __getitem__(self, idx):

        if self.is_inference:
            x = torch.FloatTensor(self.x[idx][None, ...]) / 255
            return x

        else:
            x = torch.FloatTensor(self.x[idx][None, ...]) / 255

            y = torch.tensor(self.y[idx]).type(torch.int64)
            y = F.one_hot(y, 10)
            y = y.type(torch.float32)

            return x, y

    def __len__(self):
        return len(self.x)
```

#### BERT:
```python
class TrecDataset(torch.utils.data.Dataset):
    def __init__(self, tokens, labels):
        self.tokens = tokens
        self.labels = labels

    def __getitem__(self, idx):
        input_ids = self.tokens[idx].ids
        attention_mask = self.tokens[idx].attention_mask
        labels = self.labels[idx]
        return {
            'input_ids': torch.tensor(input_ids),
            'attention_mask': torch.tensor(attention_mask),
            'labels': torch.tensor(labels)
        }

    def __len__(self):
        return len(self.labels)
```

### 2. The use of the `torch.utils.data.DataLoader` class:
* In MNIST this is automatically defined within Substra, therefore this wil ***NOT*** need to be user-defined prior to execution.

#### MNIST:
([source](custom_substra_venv/lib/python3.9/site-packages/substrafl/algorithms/pytorch/torch_base_algo.py))
```python
train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler=self._index_generator)
```

#### BERT:
```python
loader = torch.utils.data.DataLoader(
    dataset, batch_size=64
)
```

## Identifying Differences
There are several key differences between the training of pytorch MNIST and pytorch BERT, characterised by the following:

### 1. The freezing of model layers:
* In BERT, we freeze model layers prior to execution to expedite training.

#### MNIST:
```python
# NA
```

#### BERT:
```python
"""
Fine-tune the classification head only:
Freeze all BERT layer parameters, leaving just final few classification layers.
"""

for param in model.bert.parameters():
    param.requires_grad = False
```

### 2. Interim steps in the training loop:
* In BERT, we extract interim variables during the training loop.

#### MNIST:
```python
for x_batch, y_batch in train_data_loader:

    x_batch = x_batch.to(self._device)
    y_batch = y_batch.to(self._device)

    # Forward pass
    y_pred = self._model(x_batch)

    # Compute Loss
    loss = self._criterion(y_pred, y_batch)
    self._optimizer.zero_grad()
    loss.backward()
    self._optimizer.step()
```

#### BERT:
```python
for batch in loop:

    batch_mps = {
        'input_ids': batch['input_ids'].to(device),
        'attention_mask': batch['attention_mask'].to(device),
        'labels': batch['labels'].to(device)
    }

    # (Forward pass)
    # train model on batch and return outputs (incl. loss)
    outputs = model(**batch_mps)

    # (Compute Loss)
    # extract loss
    loss = outputs[0]
    # initialize calculated gradients (from prev step)
    optim.zero_grad()
    # calculate loss for every parameter that needs grad update
    loss.backward()
    # update parameters
    optim.step()
```

