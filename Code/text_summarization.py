import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, get_scheduler
from transformers import pipeline
# from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from torch.utils.data import DataLoader
import pandas as pd
import os
# from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from tqdm.auto import tqdm
from utils import avail_data, avail_models
from nltk import sent_tokenize
from accelerate import Accelerator
# import torch
import numpy as np
import random
from datasets import DatasetDict, Dataset
# import rouge_score
import evaluate

# %%--------------------------------------------------------------------------------------------------------------------
# BATCH_SIZE = 32
body_words_cutoff = 30
model_checkpoint = "JulesBelveze/t5-small-headline-generator"
TRAIN_MODEL = True
input_var = 'body'
target_var = 'headline'
splits = {'train': 0.8, 'val': 0.1, 'test': 0.1}
random_seed = 42
batch_size = 16
num_train_epochs = 25


# %%--------------------------------------------------------------------------------------------------------------------
def filter_train_indices(data):
    train_idx = [len(x) > body_words_cutoff for x in data.body.str.split()]
    # np.percentile([len(x) for x in train_val_df.body.str.split()], 1)  # 57
    return np.array(train_idx)


code_dir = os.getcwd()
data_dir = os.path.join(os.path.split(code_dir)[0], 'Data')
avail_data(data_dir)
model_dir = os.path.join(os.path.split(code_dir)[0], 'Model')
avail_models(model_dir)
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
for split, pct in splits.items():
    idx_range = (int(total_pct * len(train_val_df)), int((total_pct + pct) * len(train_val_df)))
    total_pct = total_pct + pct
    dataset[split] = Dataset.from_pandas(train_val_df.iloc[idx_range[0]:idx_range[1]])


def show_samples(data, num_samples=3, seed=random_seed):
    sample = data["train"].shuffle(seed=seed).select(range(num_samples))
    for example in sample:
        print(f"\n'>> Headline: {example[target_var]}'")
        print(f"'>> Body: {example[input_var]}'")


print("\nSample training data")
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
    label = tokenizer(
        examples[target_var], max_length=max_target_length, truncation=True
    )
    model_inputs["labels"] = label["input_ids"]
    return model_inputs


print("\nTokenizing datasets")
tokenized_datasets = dataset.map(preprocess_function, batched=True)


# %%--------------------------------------------------------------------------------------------------------------------
def one_sentence_summary(text):
    return "\n".join(sent_tokenize(text)[:1])


def evaluate_baseline(data, metric):
    summaries = [one_sentence_summary(text) for text in data[input_var]]
    return metric.compute(predictions=summaries, references=data[target_var])


# Lead-1 baseline
rouge_score = evaluate.load("rouge")
base_score = evaluate_baseline(dataset["val"], rouge_score)
print(f"\nBaseline score on validation dataset is: ")
for score, val in base_score.items():
    print(score, ": ", val)


# %%--------------------------------------------------------------------------------------------------------------------
# Training a new summarizer
# Implement a function to post-process the summaries for evaluation.
# This method is generated summaries into sentences that are separated by newlines.
# This is the format the ROUGE metric expects
def postprocess_text(preds, label):
    preds = [pred.strip() for pred in preds]
    label = [label.strip() for label in label]

    # ROUGE expects a newline after each sentence
    preds = ["\n".join(sent_tokenize(pred)) for pred in preds]
    label = ["\n".join(sent_tokenize(label)) for label in label]
    return preds, label


if TRAIN_MODEL:

    # Fine-tuning
    # Convert the dataloader into torch
    tokenized_datasets.set_format("torch")
    # Instantiate the model from cache
    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    tokenized_datasets = tokenized_datasets.remove_columns(
        dataset["train"].column_names
    )
    features = [tokenized_datasets["train"][i] for i in range(2)]
    data_collator(features)

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

    optimizer = AdamW(model.parameters(), lr=2e-5)

    # Prepare the dataset accelator prepare method
    accelerator = Accelerator()
    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader
    )

    # num_train_epochs = 5
    num_update_steps_per_epoch = len(train_dataloader)
    num_training_steps = num_train_epochs * num_update_steps_per_epoch

    # Define the learning rate schedule.
    # For the learning rate schedule, we’ll use the standard linear
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    # Save the model into 'savedmodel' folder
    output_dir = os.path.join(model_dir, "savedmodel/t5-small-headline-generator-new")
    #
    progress_bar = tqdm(range(num_training_steps))

    print("\nFine-tuning pretrained model")
    for epoch in range(num_train_epochs):
        # Training
        model.train()
        progress_bar.set_description(f"Epoch: [{epoch + 1}/{num_train_epochs}]")
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
                decoded_preds, decoded_labels = postprocess_text(
                    decoded_preds, decoded_labels
                )

                rouge_score.add_batch(predictions=decoded_preds, references=decoded_labels)

        # Compute metrics
        result = rouge_score.compute()

        # Extract the median ROUGE scores
        result = {k: round(v, 4) for k, v in result.items()}
        print(f"Epoch {epoch + 1}:", end=" ")
        for key, value in result.items():
            print(key, ' : ', value, end=" ")
        print(" ")

        # Save and upload
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(output_dir, save_function=accelerator.save)
        if accelerator.is_main_process:
            tokenizer.save_pretrained(output_dir)

# %%--------------------------------------------------------------------------------------------------------------------
# Make predictions on test data after loading model

if TRAIN_MODEL:
    savedmodel = os.path.join(model_dir, "savedmodel/t5-small-headline-generator-new")
else:
    savedmodel = os.path.join(model_dir, "savedmodel/t5-small-headline-generator")
summarizer = pipeline("summarization", model=savedmodel)


def return_summary(idx):
    reviews = dataset['test'][idx]["body"]
    titles = dataset["test"][idx]["headline"]
    summaries = summarizer(dataset["test"][idx]["body"], min_length=10, max_length=max_target_length)[0]["summary_text"]
    return reviews, titles, summaries


# Print 3 random predictions
prediction_index = []
counter = 0
while counter < 3:
    flag = False
    index = random.randint(0, len(dataset['test']))
    if index not in prediction_index:
        flag = True
        counter = counter + 1
        prediction_index.append(index)
    if flag:
        review, title, summary = return_summary(index)
        print(f'\n-- Prediction {counter} --')
        print(f"'>>> Article: {review}'")
        print(f"\n'>>> Headline: {title}'")
        print(f"\n'>>> Summary: {summary}'")


print("\n Building Predictions.csv for all test data")
main_list = []
for i in (pbar := tqdm(range(len(dataset['test'])))):
    review, title, summary = return_summary(i)
    temp = list()
    temp.append(review)
    temp.append(title)
    temp.append(summary)
    main_list.append(temp)

# Save predictions into csv file
predictions_df = pd.DataFrame(main_list, columns=['Body', 'Headline', 'Summary'])
predictions_df.to_csv(os.path.join(data_dir, 'Predictions.csv'), index=False)
