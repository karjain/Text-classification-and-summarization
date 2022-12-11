import torch
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch import nn
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader
import time

# %%------------------------------------------------------------------------------
EPOCHS = 10
LR = 0.001
BATCH_SIZE = 64
hidden_size = 16

# %%------------------------------------------------------------------------------
code_dir = os.getcwd()
data_dir = os.path.join(os.path.split(code_dir)[0], 'Data')
train_file = os.path.join(data_dir, 'Sarcasm_Headlines_Dataset.json')
test_file = os.path.join(data_dir, 'Sarcasm_Headlines_Dataset_v2.json')
df1 = pd.read_json(train_file, lines=True)
df2 = pd.read_json(test_file, lines=True)
# df_train, df_test = df1, df2
df = pd.concat([df1, df2], axis=0)
df_train, df_test = train_test_split(df, test_size=0.2, stratify=df['is_sarcastic'], shuffle=True)
train_iter = tuple(zip(list(df_train['headline']), list(df_train['is_sarcastic'])))
test_iter = tuple(zip(list(df_test['headline']), list(df_test['is_sarcastic'])))
# ------------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = get_tokenizer('basic_english')


def yield_tokens(data_iter):
    for text, _ in data_iter:
        yield tokenizer(text)


vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])
num_class = len(set([label for (text, label) in train_iter]))
vocab_size = len(vocab)


# %%------------------------------------------------------------------------------
# Glove embeddings
def get_coefs(word1, *arr):
    return word1, np.asarray(arr, dtype='float32')


embed_size = 50
embedding_file = os.path.join(data_dir, f'glove.6B.{embed_size}d.txt')
glove = dict(get_coefs(*o.rstrip().rsplit(' ')) for o in open(embedding_file))
weights_matrix = np.random.normal(scale=0.6, size=(vocab_size, embed_size))
available_words = 0
unavailable_words = 0

for word, idx in vocab.get_stoi().items():
    try:
        weights_matrix[idx] = glove[word]
        available_words += 1
    except KeyError:
        unavailable_words += 1

print(f"There are {available_words} available words and {unavailable_words} unavailable words.")


# %%------------------------------------------------------------------------------
def text_pipeline(x):
    return vocab(tokenizer(x))


def label_pipeline(y):
    # return [int(x == y) for x in range(num_class)]
    return int(y)


# %%------------------------------------------------------------------------------
def collate_batch(batch):
    label_list, text_list, offsets = [], [], [0]
    for (_text, _label) in batch:
        label_list.append(label_pipeline(_label))
        processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
        text_list.append(processed_text)
        offsets.append(processed_text.size(0))
    label_list = torch.tensor(label_list, dtype=torch.int64)
    # label_list = torch.tensor(label_list, dtype=torch.float32)
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text_list = torch.cat(text_list)
    return label_list.to(device), text_list.to(device), offsets.to(device)


# %%------------------------------------------------------------------------------
class TextClassificationModel(nn.Module):
    def __init__(self, vocabs, embed_dim, hidden_dim, num_classes, weights, trainable=False):
        super(TextClassificationModel, self).__init__()
        self.embedding = nn.EmbeddingBag(vocabs, embed_dim)
        self.embedding.load_state_dict({'weight': torch.FloatTensor(weights)})
        if trainable:
            self.embedding.weight.requires_grad = True
        else:
            self.embedding.weight.requires_grad = False
        self.lstm = nn.LSTM(embed_dim, hidden_dim)
        hs = torch.autograd.Variable(torch.zeros(1, hidden_dim))
        cs = torch.autograd.Variable(torch.zeros(1, hidden_dim))
        if device.type == 'cuda':
            self.hs = hs.cuda()
            self.cs = cs.cuda()
        else:
            self.hs = hs
            self.cs = cs
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.lstm.weight_ih_l0.data.uniform_(-initrange, initrange)
        self.lstm.weight_hh_l0.data.uniform_(-initrange, initrange)
        self.lstm.bias_ih_l0.data.uniform_(-initrange, initrange)
        self.lstm.bias_hh_l0.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        output, (self.hs, self.cs) = self.lstm(embedded, (self.hs, self.cs))
        return self.fc(output)


model = TextClassificationModel(
    vocab_size,
    embed_size,
    hidden_size,
    num_class,
    weights_matrix,
    trainable=True
).to(device)


# %%------------------------------------------------------------------------------
def train(dataloader):
    model.train()
    total_acc, total_count = 0, 0
    log_interval = 500
    start_time = time.time()

    for ix, (label, text, offsets) in enumerate(dataloader):
        optimizer.zero_grad()
        predicted_label = model(text, offsets)
        loss = criterion(predicted_label, label)
        loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        total_acc += torch.FloatTensor(predicted_label.argmax(1) == label, device=device).sum().item()
        total_count += label.size(0)
        if ix % log_interval == 0 and ix > 0:
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches '
                  '| accuracy {:8.3f}'.format(epoch, ix, len(dataloader),
                                              total_acc/total_count))
            total_acc, total_count = 0, 0
            start_time = time.time()


def evaluate(dataloader):
    model.eval()
    total_acc, total_count = 0, 0

    with torch.no_grad():
        for ix, (label, text, offsets) in enumerate(dataloader):
            predicted_label = model(text, offsets)
            # loss = criterion(predicted_label, label)
            total_acc += torch.FloatTensor(predicted_label.argmax(1) == label, device=device).sum().item()
            total_count += label.size(0)
    return total_acc/total_count


# %%------------------------------------------------------------------------------
criterion = torch.nn.CrossEntropyLoss()
# criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# %%------------------------------------------------------------------------------
train_dataloader = DataLoader(train_iter, batch_size=BATCH_SIZE,
                              shuffle=True, collate_fn=collate_batch)
test_dataloader = DataLoader(test_iter, batch_size=BATCH_SIZE,
                             shuffle=True, collate_fn=collate_batch)


for epoch in range(1, EPOCHS + 1):
    epoch_start_time = time.time()
    train(train_dataloader)
    accu_val = evaluate(test_dataloader)
    print('-' * 59)
    print('| end of epoch {:3d} | time: {:5.2f}s | '
          'valid accuracy {:8.3f} '.format(epoch,
                                           time.time() - epoch_start_time, accu_val))
    print('-' * 59)
# %%------------------------------------------------------------------------------
torch.save(model.state_dict(), 'model_weights.pt')
# load the model first for inference
model.load_state_dict(torch.load('model_weights.pt'))
model.eval()
