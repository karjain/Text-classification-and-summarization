from torch import nn
from transformers import RobertaTokenizerFast, RobertaModel
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.optim import Adam
from tqdm import tqdm
import re
import nltk
from nltk.corpus import stopwords
import numpy as np
import matplotlib.pyplot as plt
from utils import *
import argparse

plt.style.use('ggplot')
BATCH_SIZE = 32
model_save_name = 'trans-roberta-model.pt'

parser = argparse.ArgumentParser()
parser.add_argument('-Train',  action='store_true')
args = parser.parse_args()

option = args.Train
print(f'option={option}')
if args.Train:
    TRAIN_MODEL = True
    print('Training')
else:
    print('only predict')
    TRAIN_MODEL = False

print(f'TRAIN_MODEL={TRAIN_MODEL}')

code_dir = os.getcwd()
data_dir = os.path.join(os.path.split(code_dir)[0], 'Data')
avail_data(data_dir)
model_dir = os.path.join(os.path.split(code_dir)[0], 'Model')
avail_models(model_dir)
final_data = pd.read_json(os.path.join(data_dir, r'Combined_Headlines.json'))
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

tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')

MAX_LEN = max([len(sent.split()) for sent in final_df['text']])
NUM_CLASSES = final_df['label'].nunique()
labels = {
    0: 0,
    1: 1
}


class TextLabelDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.labels = [[labels[1-label], labels[label]] for label in df['label']]
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
    final_df.sample(frac=1),
    test_size=0.1,
    stratify=final_df['label']
)
df_train, df_val = train_test_split(
    df_train_val.sample(frac=1),
    test_size=0.15,
    stratify=df_train_val['label']
)


class TransformerClassifier(nn.Module):

    def __init__(self, dropout=0.2):
        super(TransformerClassifier, self).__init__()
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
        encoded_layers = self.trans(input_ids=input_id, attention_mask=mask, return_dict=False)[0]
        encoded_layers = encoded_layers.permute(0, 2, 1)
        maxpool_output = self.maxpool(encoded_layers).squeeze(2)
        linear_output = self.linear(maxpool_output)
        linear_output = self.log_softmax(linear_output)
        return linear_output


def train_model(model, train_data, val_data, learning_rate, epochs):
    train, val = TextLabelDataset(train_data), TextLabelDataset(val_data)
    train_dataloader = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val, batch_size=BATCH_SIZE)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Training on {device}")
    criterion = nn.KLDivLoss(reduction="batchmean")
    optimizer = Adam(model.parameters(), lr=learning_rate)
    save_best_model = SaveBestModel()
    acc_train_e = list()
    loss_train_e = list()
    acc_valid_e = list()
    loss_valid_e = list()
    epoch_list = list()

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

        acc_train_e.append(total_acc_train / len(train_data))
        loss_train_e.append(total_loss_train / len(train_data))
        acc_valid_e.append(total_acc_val / len(val_data))
        loss_valid_e.append(total_loss_val / len(val_data))
        epoch_list.append(epoch_num + 1)
        metric_df = pd.DataFrame(list(zip(epoch_list, acc_train_e, loss_train_e, acc_valid_e, loss_valid_e)),
                                 columns=['Epoch', 'Accuracy_Train', 'Loss_train', 'Accuracy_valid', 'Loss_valid'])
        save_best_model(total_loss_val / len(val_data), epoch_num + 1, model, model_save_name)

        # print(metric_df)
        if epoch_num == epochs-1:
            metric_df.to_csv(os.path.join(data_dir, 'Metrics.csv'))
            plt.figure(figsize=(10, 10))
            plt.plot(metric_df['Epoch'], metric_df['Accuracy_Train'], label='Train Accuracy')
            plt.plot(metric_df['Epoch'], metric_df['Accuracy_valid'], label='Validation Accuracy')
            plt.legend(fontsize=10)
            plt.title('Model Performance per Epoch', fontsize=20)
            plt.xlabel('Epoch', fontsize=15)
            plt.ylabel('Accuracy', fontsize=15)
            plt.savefig(os.path.join(data_dir, 'Accuracy_plot.png'))

            plt.figure(figsize=(10, 10))
            plt.plot(metric_df['Epoch'], metric_df['Loss_train'], label='Train Loss')
            plt.plot(metric_df['Epoch'], metric_df['Loss_valid'], label='Validation Loss')
            plt.legend(fontsize=10)
            plt.title('Model Performance per Epoch', fontsize=20)
            plt.xlabel('Epoch', fontsize=15)
            plt.ylabel('Loss', fontsize=15)

            plt.savefig(os.path.join(data_dir, 'Loss_plot.png'))
            # plt.show()


EPOCHS = 6
tf_model = TransformerClassifier()
LR = 1e-5

if TRAIN_MODEL:
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
    best_model.load_state_dict(torch.load(os.path.join(model_dir, model_save_name)))
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
