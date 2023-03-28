from timeit import timeit
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.has_mps)
print(device)

trec = load_dataset("trec", split='train[:1000]')
print(trec)

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

text = trec['text'][:64]
tokens = tokenizer(
    text, max_length=512,
    truncation = True, padding=True,
    return_tensors='pt'
)

device = torch.device('mps')
model.to(device)
tokens.to(device)
print(device)


import numpy as np

labels = np.zeros(
    (len(trec), max(trec['label-coarse'])+1)
)
# one-hot encode
labels[np.arange(len(trec)), trec['label-coarse']] = 1
print(labels[:5])

class TrecDataset(torch.utils.data.Dataset):
    def __init__(self, tokens, labels):
        self.tokens = tokens
        self.labels = labels

    def __getitem__(self, idx):
        input_ids = self.tokens[idx].ids
        attention_mask = self.tokens[idx].attention_mask
        labels = self.labels[idx]
        return{
            'input_ids': torch.tensor(input_ids),
            'attention_mask': torch.tensor(attention_mask),
            'labels': torch.tensor(labels)
        }

    def __len__(self):
        return len(self.labels)

dataset = TrecDataset(tokens, labels)

loader = torch.utils.data.DataLoader(
    dataset, batch_size=1
)


from transformers import BertForSequenceClassification, BertConfig

config = BertConfig.from_pretrained('bert-base-uncased')
config.num_labels = max(trec['label-coarse'])+1  # create 6 outputs
model = BertForSequenceClassification(config).to(device)


# activate training mode of model
model.train()

# initialize adam optimizer
optim = torch.optim.Adam(model.parameters(), lr=5e-5)

from time import time
from tqdm.auto import tqdm

loop_time = []
model.to(device)

# setup loop
loop = tqdm(loader, leave=True)
for batch in loop:
    batch_mps = {
        'input_ids': batch['input_ids'].to(device),
        'attention_mask': batch['attention_mask'].to(device),
        'labels': batch['labels'].type(torch.float).to(device)
    }
    t0 = time()

    optim.zero_grad()
    outputs = model(**batch_mps)
    # extract loss
    loss = outputs[0]
    # calculate loss
    loss.backward()
    # update params
    optim.step()
    loop_time.append(time()-t0)
    # print relevant info to progress bar
    loop.set_postfix(loss=loss.item())

    


# # begin training loop
# for batch in loader:
#   	# note that we move everything to the MPS device
#     batch_mps = {
#         'input_ids': batch['input_ids'].to(device),
#         'attention_mask': batch['attention_mask'].to(device),
#         'labels': batch['labels'].to(device)
#     }
#     # initialize calculated gradients (from prev step)
#     optim.zero_grad()
#     # train model on batch and return outputs (incl. loss)
#     outputs = model(**batch_mps)
#     # extract loss
#     loss = outputs[0]
#     # calculate loss for every parameter that needs grad update
#     loss.backward()
#     # update parameters
#     optim.step()


