# Imports
import pandas as pd
import re
import nltk
# nltk.download()
from nltk.corpus import stopwords
import torch
import numpy as np
from transformers import BertTokenizer
from torch import nn
from transformers import BertModel
from torch.optim import AdamW
from tqdm import tqdm
import sys
import os

code_dir = os.getcwd()
data_dir = os.path.join(os.path.split(code_dir)[0], 'Data')
utils_dir = os.path.join(os.path.split(code_dir)[0], 'Code')
sys.path.insert(0, f'{utils_dir}')
from utils import avail_data
avail_data(data_dir)
model_dir = os.path.join(os.path.split(code_dir)[0], 'Model')
if not os.path.exists(model_dir):
    os.mkdir(model_dir)

data = pd.read_json(os.path.join(data_dir,'Sarcasm_Headlines_Dataset.json'), lines = True)
data_v2 = pd.read_json(os.path.join(data_dir,'Sarcasm_Headlines_Dataset.json'), lines = True)

print(data.shape)
print(data_v2.shape)

final_data = pd.concat([data ,data_v2] , ignore_index = True , axis =0)
print(final_data.shape)
final_data.info()

# Final dataset
df = pd.DataFrame()
df['text'] = final_data['headline']
df['label'] = final_data['is_sarcastic']


# # pre processing

# remove numbers
print("Removing Numbers from text")
df['text'] = [re.sub(r'[^a-zA-Z]', ' ', str(sentence)) for sentence in df['text']]
print(df.head())

# lower the characters
print("Lowering Characters")
df['text'] = [sentence.lower() for sentence in df['text']]
print(df.head())

# remove stopwords
print("Removing Stopwords from text")
list =[]
for sentence in df['text']:
    list.append([i for i in str(sentence).split() if i not in (stopwords.words('english'))])

df['text'] = [' '.join(l) for l in list]
print(df.head())

# lemmatization
print("Lemmatizing text")
wnl = nltk.WordNetLemmatizer()
list = []
for sentence in df['text']:
    list.append(' '.join(wnl.lemmatize(word) for word in sentence.split()))

df['text'] = list
print(df.head())

# BERT Tockenizing
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
labels = {0: 0,
          1: 1
          }

class Dataset(torch.utils.data.Dataset):

    def __init__(self, df):
        self.labels = [labels[label] for label in df['label']]
        # self.labels = labels
        self.texts = [tokenizer(text,
                                padding='max_length', max_length=256, truncation=True,
                                return_tensors="pt") for text in df['text']]

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self.texts[idx]

    def __getitem__(self, idx):
        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)

        return batch_texts, batch_y


# Train-Test-Validation split
np.random.seed(112)
df_train, df_val, df_test = np.split(df.sample(frac=1, random_state=42),
                                     [int(.8 * len(df)), int(.9 * len(df))])

print(len(df_train), len(df_val), len(df_test))

# Model Definition
class BertClassifier(nn.Module):

    def __init__(self, dropout=0.3):
        super(BertClassifier, self).__init__()

        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(768, 384)
        self.gelu1 = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(384, 2)
        self.gelu2 = nn.GELU()

    def forward(self, input_id, mask):
        _, pooled_output = self.bert(input_ids=input_id, attention_mask=mask, return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output1 = self.linear1(dropout_output)
        first_layer = self.dropout(self.gelu1(linear_output1))
        linear_output = self.linear2(first_layer)
        final_layer = self.gelu2(linear_output)

        return final_layer


#  Training & Validating
def train(model, train_data, val_data, learning_rate, epochs):
    train, val = Dataset(train_data), Dataset(val_data)

    train_dataloader = torch.utils.data.DataLoader(train, batch_size=8, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=8)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Training on {device}")
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    if use_cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    for epoch_num in range(epochs):

        total_acc_train = 0
        total_loss_train = 0

        for train_input, train_label in tqdm(train_dataloader):
            train_label = train_label.to(device)
            mask = train_input['attention_mask'].to(device)
            input_id = train_input['input_ids'].squeeze(1).to(device)

            output = model(input_id, mask)

            batch_loss = criterion(output, train_label.long())
            total_loss_train += batch_loss.item()

            acc = (output.argmax(dim=1) == train_label).sum().item()
            total_acc_train += acc

            model.zero_grad()
            batch_loss.backward()
            optimizer.step()

        total_acc_val = 0
        total_loss_val = 0

        with torch.no_grad():

            for val_input, val_label in val_dataloader:
                val_label = val_label.to(device)
                mask = val_input['attention_mask'].to(device)
                input_id = val_input['input_ids'].squeeze(1).to(device)

                output = model(input_id, mask)

                batch_loss = criterion(output, val_label.long())
                total_loss_val += batch_loss.item()

                acc = (output.argmax(dim=1) == val_label).sum().item()
                total_acc_val += acc

        print(
            f'Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / len(train_data): .3f} \
                | Train Accuracy: {total_acc_train / len(train_data): .3f} \
                | Val Loss: {total_loss_val / len(val_data): .3f} \
                | Val Accuracy: {total_acc_val / len(val_data): .3f}')

EPOCHS = 4
model = BertClassifier()
LR = 1e-6

train(model, df_train, df_val, LR, EPOCHS)

# Testing
def evaluate(model, test_data):
    test = Dataset(test_data)

    test_dataloader = torch.utils.data.DataLoader(test, batch_size=8)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Testing on {device}")
    if use_cuda:
        model = model.cuda()

    total_acc_test = 0
    with torch.no_grad():

        for test_input, test_label in test_dataloader:
            test_label = test_label.to(device)
            mask = test_input['attention_mask'].to(device)
            input_id = test_input['input_ids'].squeeze(1).to(device)

            output = model(input_id, mask)

            acc = (output.argmax(dim=1) == test_label).sum().item()
            total_acc_test += acc

    print(f'Test Accuracy: {total_acc_test / len(test_data): .3f}')

evaluate(model, df_test)

