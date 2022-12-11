from torch import nn
# from tokenizers.implementations import ByteLevelBPETokenizer
# from transformers import BertModel, BertTokenizer, DistilBertTokenizerFast, DistilBertModel
from transformers import RobertaTokenizerFast, RobertaModel
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from torch.optim import Adam
from tqdm import tqdm
import re
import nltk
from nltk.corpus import stopwords
import torch
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
BATCH_SIZE = 32

code_dir = os.getcwd()
data_dir = os.path.join(os.path.split(code_dir)[0], 'Data')
data = pd.read_json(os.path.join(data_dir, r'Sarcasm_Headlines_Dataset.json'), lines=True)
data_v2 = pd.read_json(os.path.join(data_dir, r'Sarcasm_Headlines_Dataset_v2.json'), lines=True)

final_data = pd.concat([data, data_v2], ignore_index=True, axis=0)
final_data.info()

# Final dataset
final_df = pd.DataFrame()
final_df['text'] = final_data['headline']
final_df['label'] = final_data['is_sarcastic']
final_df.groupby(['label']).size().plot.bar()


# # pre processing

# remove numbers
final_df['text'] = [re.sub(r'[^a-zA-Z]', ' ', str(sentence)) for sentence in final_df['text']]

# lower the characters
final_df['text'] = [sentence.lower() for sentence in final_df['text']]

# remove stopwords
lst = []
for sentence in final_df['text']:
    lst.append([i for i in str(sentence).split() if i not in (stopwords.words('english'))])

final_df['text'] = [' '.join(sent) for sent in lst]

# lemmatization
wnl = nltk.WordNetLemmatizer()
lst = []
for sentence in final_df['text']:
    lst.append(' '.join(wnl.lemmatize(word) for word in sentence.split()))

final_df['text'] = lst

# trained_tokenizer = ByteLevelBPETokenizer()
# text_file = os.path.join(data_dir, "tokenizer.txt")
# with open(text_file, 'w') as f:
#     for line in final_df['text']:
#         f.write(line)
#         f.write('\n')
#
# trained_tokenizer.train(
#     files= text_file,
#     vocab_size=52_000,
#     min_frequency=2,
#     special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
# )
# trained_tokenizer.save_model(data_dir)
# trained_tokenizer = ByteLevelBPETokenizer(os.path.join(data_dir, "vocab.json"), os.path.join(data_dir, "merges.txt"),)

# tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
# tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-cased')
tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
# tokenizer = RobertaTokenizerFast.from_pretrained(data_dir)

# labels = [0,1]
MAX_LEN = max([len(sent.split()) for sent in final_df['text']])
NUM_CLASSES = final_df['label'].nunique()
labels = {
    0: 0,
    1: 1
}


class TextLabelDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        # self.labels = [labels[label] for label in df['label']]
        self.labels = [[labels[1-label], labels[label]] for label in df['label']]
        # self.labels = labels
        self.texts = [tokenizer(text,
                                padding='max_length', max_length=MAX_LEN, truncation=True,
                                return_tensors="pt") for text in df['text']]

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


df_train_val, df_test = train_test_split(
    final_df.sample(frac=1, random_state=42),
    test_size=0.1,
    stratify=final_df['label']
)
df_train, df_val = train_test_split(
    df_train_val.sample(frac=1, random_state=42),
    test_size=0.15,
    stratify=df_train_val['label']
)


class SaveBestModel:
    def __init__(self, best_valid_loss=float('inf')):
        self.best_valid_loss = best_valid_loss
        self.model_dir = data_dir
        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)

    def __call__(self, current_valid_loss, epoch, model):
        if current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            print(f"Best validation loss: {self.best_valid_loss}")
            print(f"Saving best model for epoch: {epoch}\n")
            torch.save(
                model.state_dict(),
                os.path.join(self.model_dir, 'trans-model.pt')
            )


class TransformerClassifier(nn.Module):

    def __init__(self, dropout=0.2):
        super(TransformerClassifier, self).__init__()
        # self.trans = BertModel.from_pretrained('bert-base-cased')
        # self.trans = DistilBertModel.from_pretrained('distilbert-base-cased')
        self.trans = RobertaModel.from_pretrained('roberta-base')
        self.hidden_size = self.trans.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size, bidirectional=True)
        self.maxpool = nn.MaxPool1d(MAX_LEN)
        self.linear = nn.Linear(self.hidden_size, 2)
        self.gelu = nn.GELU()
        self.softmax = nn.Softmax(dim=1)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, input_id, mask):
        # encoded_layers, pooled_output = self.trans(input_ids=input_id, attention_mask=mask, return_dict=False)
        encoded_layers = self.trans(input_ids=input_id, attention_mask=mask, return_dict=False)[0]
        # output = self.dropout(output)
        # output = output.permute(1, 0, 2)
        # output, (hs, cs) = self.lstm(output)
        encoded_layers = encoded_layers.permute(0, 2, 1)
        maxpool_output = self.maxpool(encoded_layers).squeeze(2)
        # lstm_output, _ = self.lstm(maxpool_output)
        linear_output = self.linear(maxpool_output)
        # linear_output = self.gelu(linear_output)
        # linear_output = self.softmax(linear_output)
        linear_output = self.log_softmax(linear_output)
        return linear_output


def train_model(model, train_data, val_data, learning_rate, epochs):
    train, val = TextLabelDataset(train_data), TextLabelDataset(val_data)
    train_dataloader = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val, batch_size=BATCH_SIZE)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Training on {device}")
    # criterion = nn.CrossEntropyLoss()
    # criterion = nn.BCEWithLogitsLoss()
    criterion = nn.KLDivLoss(reduction="batchmean")
    optimizer = Adam(model.parameters(), lr=learning_rate)
    save_best_model = SaveBestModel()
    if use_cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    for epoch_num in range(epochs):
        total_acc_train = 0
        total_loss_train = 0
        total_len_train = 0
        for train_input, train_label in (pbar := tqdm(train_dataloader)):
            train_label = train_label.to(device)
            mask = train_input['attention_mask'].to(device)
            input_id = train_input['input_ids'].squeeze(1).to(device)
            output = model(input_id, mask)
            batch_loss = criterion(output, train_label.float())
            total_loss_train += batch_loss.item()
            acc = torch.logical_not(torch.logical_xor(output.argmax(dim=1), train_label.argmax(dim=1))).sum().item()
            total_acc_train += acc
            total_len_train += len(train_label)
            model.zero_grad()
            batch_loss.backward()
            optimizer.step()
            pbar.set_postfix({
                "train_loss": total_loss_train / total_len_train,
                "train_acc": total_acc_train / total_len_train
            })
        total_acc_val = 0
        total_loss_val = 0
        total_len_val = 0

        with torch.no_grad():
            for val_input, val_label in (pbar := tqdm(val_dataloader)):
                val_label = val_label.to(device)
                mask, input_id = val_input['attention_mask'].to(device), val_input['input_ids'].squeeze(1).to(device)
                output = model(input_id, mask)
                batch_loss = criterion(output, val_label.float())
                total_loss_val += batch_loss.item()
                acc = torch.logical_not(torch.logical_xor(output.argmax(dim=1), val_label.argmax(dim=1))).sum()
                total_acc_val += acc.item()
                total_len_val += len(val_label)
                pbar.set_postfix({
                    "val_loss": total_loss_val / total_len_val,
                    "val_acc": total_acc_val / total_len_val
                })
        print(f"Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / len(train_data): .3f} | "
              f"Train Accuracy: {total_acc_train / len(train_data): .3f} | "
              f"Val Loss: {total_loss_val / len(val_data): .3f} | "
              f"Val Accuracy: {total_acc_val / len(val_data): .3f}")
        save_best_model(total_loss_val / len(val_data), epoch_num + 1, model)


EPOCHS = 5
tf_model = TransformerClassifier()
LR = 1e-5

train_model(tf_model, df_train, df_val, LR, EPOCHS)


def evaluate(test_data):
    test = TextLabelDataset(test_data)
    test_dataloader = DataLoader(test, batch_size=BATCH_SIZE)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Testing on {device}")
    best_model = TransformerClassifier()
    if use_cuda:
        best_model = best_model.cuda()
    best_model.load_state_dict(torch.load(os.path.join(data_dir, 'trans-model.pt')))
    total_acc_test = 0
    total_len_test = 0
    with torch.no_grad():
        for test_input, test_label in (pbar := tqdm(test_dataloader)):
            test_label = test_label.to(device)
            mask = test_input['attention_mask'].to(device)
            input_id = test_input['input_ids'].squeeze(1).to(device)
            output = best_model(input_id, mask)
            acc = torch.logical_not(torch.logical_xor(output.argmax(dim=1), test_label.argmax(dim=1))).sum().item()
            total_acc_test += acc
            total_len_test += len(test_label)
            pbar.set_postfix({"test_acc": total_acc_test / total_len_test})
    print(f'Test Accuracy: {total_acc_test / len(test_data): .3f}')


evaluate(df_test)
