# from torch import nn
# from transformers import BartForConditionalGeneration, BartTokenizerFast
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments
# from transformers import Seq2SeqTrainer, DataCollatorForSeq2Seq
# from torch.utils.data import DataLoader
import nltk
import pandas as pd
import os
from datasets import Dataset, DatasetDict
from nltk import sent_tokenize
# from datasets import DatasetDict, Dataset
# import rouge_score
import evaluate
from torch.utils.data import DataLoader
from torch.optim import AdamW
from accelerate import Accelerator
from transformers import get_scheduler
from tqdm.auto import tqdm
import torch
import numpy as np

# %%--------------------------------------------------------------------------------------------------------------------
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq, Seq2SeqTrainer, \
    AutoTokenizer

# BATCH_SIZE = 32
body_words_cutoff = 30
model_checkpoint = "JulesBelveze/t5-small-headline-generator"
input_var = 'body'
target_var = 'headline'
splits = {'train': 0.8, 'val': 0.1, 'test': 0.1}
random_seed = 42
batch_size = 16
num_train_epochs = 16


# %%--------------------------------------------------------------------------------------------------------------------
def filter_train_indices(data):
    train_idx = [len(x) > body_words_cutoff for x in data.body.str.split()]
    # np.percentile([len(x) for x in train_val_df.body.str.split()], 1)  # 57
    return np.array(train_idx)


code_dir = os.getcwd()
data_dir = os.path.join(os.path.split(code_dir)[0], '')
df = pd.read_json('Data/sarcastic_output.json')
# df = df[:50]
# print(df.columns)
df = df[['body', 'headline']]
train_val_df = df[filter_train_indices(df)]
test_df = df[~filter_train_indices(df)]
max_input_length = int(np.percentile([len(x) for x in train_val_df[input_var].str.split()], 99.5))
max_target_length = int(np.percentile([len(x) for x in train_val_df[target_var].str.split()], 99.5))
# max_target_length = 10
print(max_target_length)
# %%--------------------------------------------------------------------------------------------------------------------
dataset = DatasetDict()
train_val_df = train_val_df.sample(frac=1, random_state=random_seed).reset_index()
total_pct = 0
for split, pct in splits.items():
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
# Fine-tuning
# Convert the dataloader into torch

tokenized_datasets.set_format("torch")
# Insantiate the model from cache
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
tokenized_datasets = tokenized_datasets.remove_columns(
    dataset["train"].column_names
)
features = [tokenized_datasets["train"][i] for i in range(2)]
data_collator(features)
rouge_score = evaluate.load("rouge")
# Define our own dataloader
# batch_size = 8
train_dataloader = DataLoader(
    tokenized_datasets["train"],
    shuffle=True,
    collate_fn=data_collator,
    batch_size=batch_size,
)
eval_dataloader = DataLoader(
    tokenized_datasets["val"], collate_fn=data_collator, batch_size=batch_size
)

# Define the optimizer AdamW
optimizer = AdamW(model.parameters(), lr=2e-5)

# Now Prepare the dataset accelator prepare method

accelerator = Accelerator()
model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader, eval_dataloader
)

# Define the learning rate schedule.
# Implement a function to post-process the summaries for evaluation.
# Create a repository on the Hub that we can push our model to.
# For the learning rate schedule, we’ll use the standard linear

# num_train_epochs = 5
num_update_steps_per_epoch = len(train_dataloader)
num_training_steps = num_train_epochs * num_update_steps_per_epoch

lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)


# This method is generated summaries into sentences that are sperated by newlines.
# This is the format the ROUGE metric expects

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # ROUGE expects a newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels


# Save the model into 'savedmodel' folder
output_dir = 'savedmodel/small-t5'
#
progress_bar = tqdm(range(num_training_steps))

for epoch in range(num_train_epochs):
    # Training
    model.train()
    for step, batch in enumerate(train_dataloader):
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

    # Evaluation
    model.eval()
    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            generated_tokens = accelerator.unwrap_model(model).generate(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )

            generated_tokens = accelerator.pad_across_processes(
                generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
            )
            labels = batch["labels"]

            # If we did not pad to max length, we need to pad the labels too
            labels = accelerator.pad_across_processes(
                batch["labels"], dim=1, pad_index=tokenizer.pad_token_id
            )

            generated_tokens = accelerator.gather(generated_tokens).cpu().numpy()
            labels = accelerator.gather(labels).cpu().numpy()

            # Replace -100 in the labels as we can't decode them
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
            if isinstance(generated_tokens, tuple):
                generated_tokens = generated_tokens[0]
            decoded_preds = tokenizer.batch_decode(
                generated_tokens, skip_special_tokens=True
            )

            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
            # print(decoded_preds)
            # print(decoded_labels)
            decoded_preds, decoded_labels = postprocess_text(
                decoded_preds, decoded_labels
            )

            rouge_score.add_batch(predictions=decoded_preds, references=decoded_labels)

    # Compute metrics
    result = rouge_score.compute()
    # print(result.items())
    # Extract the median ROUGE scores
    # result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
    result = {k: round(v, 4) for k, v in result.items()}
    print(f"Epoch {epoch}:", result)

    # Save and upload
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(output_dir, save_function=accelerator.save)
    if accelerator.is_main_process:
        tokenizer.save_pretrained(output_dir)

from transformers import pipeline

savedmodel = "savedmodel/small-t5"
summarizer = pipeline("summarization", model=savedmodel)

# print(dataset['test'][1])


def print_summary(idx):
    print(f'\n-- Prediction {i} --')
    review = dataset['test'][idx]["body"]
    title = dataset["test"][idx]["headline"]
    summary = summarizer(dataset["test"][idx]["body"],min_length=10, max_length=max_target_length)[0]["summary_text"]
    print(f"'>>> Article: {review}'")
    print(f"\n'>>> Headline: {title}'")
    print(f"\n'>>> Summary: {summary}'")


for i in range(len(dataset['test'][:3])):
    print_summary(i)