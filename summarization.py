# from torch import nn
# from transformers import BartForConditionalGeneration, BartTokenizerFast
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments
from transformers import Seq2SeqTrainer, DataCollatorForSeq2Seq
# from torch.utils.data import DataLoader
import pandas as pd
import os
# from sklearn.model_selection import train_test_split
# from torch.optim import AdamW
# from tqdm import tqdm
from nltk import sent_tokenize
# import torch
import numpy as np
from datasets import DatasetDict, Dataset
# import rouge_score
import evaluate

# %%--------------------------------------------------------------------------------------------------------------------
BATCH_SIZE = 32
body_words_cutoff = 30
model_checkpoint = "t5-small"
input_var = 'body'
target_var = 'headline'
splits = {'train': 0.8, 'val': 0.1, 'test': 0.1}
random_seed = 42
batch_size = 16
num_train_epochs = 8


# %%--------------------------------------------------------------------------------------------------------------------
def filter_train_indices(data):
    train_idx = [len(x) > body_words_cutoff for x in data.body.str.split()]
    # np.percentile([len(x) for x in train_val_df.body.str.split()], 1)  # 57
    return np.array(train_idx)


code_dir = os.getcwd()
data_dir = os.path.join(os.path.split(code_dir)[0], 'Data')
df = pd.read_json(os.path.join(data_dir, 'sarcastic_output.json'))
df = df[['body', 'headline']]
train_val_df = df[filter_train_indices(df)]
test_df = df[~filter_train_indices(df)]
max_input_length = int(np.percentile([len(x) for x in train_val_df[input_var].str.split()], 99.5))
max_target_length = int(np.percentile([len(x) for x in train_val_df[target_var].str.split()], 99.5))


# %%--------------------------------------------------------------------------------------------------------------------
dataset = DatasetDict()
train_val_df = train_val_df.sample(frac=1, random_state=random_seed).reset_index()
total_pct = 0
for split, pct  in splits.items():
    idx_range = (int(total_pct * len(train_val_df)), int((total_pct + pct) * len(train_val_df)))
    total_pct = total_pct + pct
    dataset[split] = Dataset.from_pandas(train_val_df.iloc[idx_range[0]:idx_range[1]])


def show_samples(data, num_samples=3, seed=random_seed):
    sample = data["train"].shuffle(seed=seed).select(range(num_samples))
    for example in sample:
        print(f"\n'>> Headline: {example[target_var]}'")
        print(f"'>> Body: {example[input_var]}'")


show_samples(dataset)
# %%--------------------------------------------------------------------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
tokenized = tokenizer("I love Natural Language Processing!")
# print(tokenizer.convert_ids_to_tokens(tokenized.input_ids))


# %%--------------------------------------------------------------------------------------------------------------------
def preprocess_function(examples):
    model_inputs = tokenizer(
        examples[input_var],
        max_length=max_input_length,
        truncation=True
    )
    labels = tokenizer(
        examples[target_var], max_length=max_target_length, truncation=True
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


print("Tokenizing datasets")
tokenized_datasets = dataset.map(preprocess_function, batched=True)


# %%--------------------------------------------------------------------------------------------------------------------
def one_sentence_summary(text):
    return "\n".join(sent_tokenize(text)[:1])


def evaluate_baseline(data, metric):
    summaries = [one_sentence_summary(text) for text in data[input_var]]
    return metric.compute(predictions=summaries, references=data[target_var])


# Lead-1 baseline
rouge_score = evaluate.load("rouge")
print(one_sentence_summary(dataset["train"][1][target_var]))
base_score = evaluate_baseline(dataset["val"], rouge_score)
print(f"Baseline score on validation dataset is: ")
for score, val in base_score.items():
    print(score, ": ", val)

# %%--------------------------------------------------------------------------------------------------------------------
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
logging_steps = len(tokenized_datasets["train"]) // batch_size
model_name = model_checkpoint.split("/")[-1]

args = Seq2SeqTrainingArguments(
    output_dir=os.path.join(data_dir, f"{model_name}-finetuned-on-sarcastic-news"),
    evaluation_strategy="epoch",
    learning_rate=5.6e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=num_train_epochs,
    predict_with_generate=True,
    logging_steps=logging_steps,
    push_to_hub=False
)


# %%--------------------------------------------------------------------------------------------------------------------
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    # Decode generated summaries into text
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    # Decode reference summaries into text
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    # ROUGE expects a newline after each sentence
    decoded_preds = ["\n".join(sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(sent_tokenize(label.strip())) for label in decoded_labels]
    # Compute ROUGE scores
    result = rouge_score.compute(
        predictions=decoded_preds, references=decoded_labels, use_stemmer=True
    )
    # Extract the median scores
    result = {key: round(value * 100, 4) for key, value in result.items()}
    return result


data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
tokenized_datasets = tokenized_datasets.remove_columns(
    dataset["train"].column_names
)
features = [tokenized_datasets["train"][i] for i in range(2)]
data_collator(features)

# %%--------------------------------------------------------------------------------------------------------------------
trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["val"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)
trainer.train()
