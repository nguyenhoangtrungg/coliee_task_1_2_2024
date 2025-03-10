from core.bm25.run_bm25 import *
from core.model.t5_model_dataset import * 
from core.weak_label import create_weak_dataset


import wandb
wandb.login()

import torch
import torch.nn as nn
from transformers import  AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from torch.utils.data import DataLoader
from constant import *
 
import nltk
import evaluate
import numpy as np

print(args)
nltk.download("punkt", quiet=True)
metric = evaluate.load("exact_match")

os.environ["WANDB_PROJECT"] = "2025_collie_task_2"

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForSeq2SeqLM.from_pretrained(PRETRAIN_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    model.to(device)

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        # print("Preds: ", preds, "Labels: ", labels)

        # decode preds and labels
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        # print("Pred: ", decoded_preds, "Label: ", decoded_labels)

        # rougeLSum expects newline after each sentence
        decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
        decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]
        print("Pred: ", decoded_preds[:10], "Label: ", decoded_labels[:10])
        result = metric.compute(predictions=decoded_preds, references=decoded_labels)

        num_label = 0
        num_true_label = 0
        num_predict = 0
        error_rate = 0
        for i in range(len(decoded_preds)):
            pred = decoded_preds[i]
            label = decoded_labels[i]
            if label == 'true':
                num_label += 1
            if label == 'true' and pred == 'true':
                num_true_label += 1
            if pred == 'true':
                num_predict += 1

            if label not in ['true', 'false']:
                error_rate += 1
        precision = num_true_label / num_predict if num_predict > 0 else 0
        recall = num_true_label / num_label if num_label > 0 else 0
        try:
            f1 = 2 * precision * recall / (precision + recall)
        except:
            f1 = 0
        result['f1'] = f1
        result['precision'] = precision
        result['recall'] = recall
        result['error_rate'] = error_rate
        
        return result
    
    def smart_batching_collate_text_only(batch):
        texts = [example['text'] for example in batch]
        tokenized = tokenizer(texts, padding=True, truncation='longest_first', return_tensors='pt', max_length=512)
        tokenized['labels'] = tokenizer([example['labels'] for example in batch], return_tensors='pt')['input_ids']

        for name in tokenized:
            tokenized[name] = tokenized[name].to(device)

        return tokenized
    


    promt = "Recognizing entailment between a decision fragment and a relevant legal paragraph."

    # DATA LOADING
    if CHOOSE_WEAK == 0:
        train_df = run_create_csv_bm25(TRAINING_PATH, TRAIN_LABEL_PATH, CSV_TRAINING_DATA_PATH, "train", NEGATIVE_MODE, NEGATIVE_NUM, "t5")

    elif CHOOSE_WEAK == 1:
        train_df = create_weak_dataset.create_weak_dataset(WEAK_DATASET_PATH, CSV_WEAK_DATA_PATH, MIN_LEN, MAX_LEN, "t5", banlance=0)

    elif CHOOSE_WEAK == 2:
        train_df = create_weak_dataset.create_weak_dataset(WEAK_DATASET_PATH, CSV_WEAK_DATA_PATH, MIN_LEN, MAX_LEN, "t5", banlance=1)
    valid_df = run_create_csv_bm25(TESTING_PATH, TEST_LABEL_PATH, CSV_TESTING_DATA_PATH, "test", NEGATIVE_MODE, NEGATIVE_NUM, "t5")

    if FAST_DEV_RUN == "1":
        train_df = train_df[:5]
        valid_df = valid_df[:5]

    # print(valid_df["label"][:10])

    print("Train data len: ", len(train_df))
    print(train_df.head())

    print("Valid data len: ", len(valid_df))
    print(valid_df.head())


    dataset_train = MonoT5Dataset(train_df["fragment"].tolist(), train_df["content"].tolist(), train_df["label"].tolist())
    dataset_test = MonoT5Dataset(valid_df["fragment"].tolist(), valid_df["content"].tolist(), valid_df["label"].tolist())


    class CustomTrainer(Seq2SeqTrainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            labels = inputs.pop("labels")
            inputs["decoder_input_ids"] = inputs["input_ids"].clone()
            print(inputs)
            print(labels)
            outputs = model(**inputs)
            logits = outputs.logits

            # Chúng ta chỉ quan tâm đến token đầu tiên trong output vì đó là điểm relevance
            logits = logits[:, 0, :]  # Chọn token đầu tiên
            labels = labels[:, 0]  # Chỉ giữ lại token đầu tiên của labels
            
            # Convert logits thành xác suất với softmax
            probs = nn.functional.log_softmax(logits, dim=-1)
            
            # Cross-entropy loss
            loss = nn.functional.nll_loss(probs, labels)
            print("Lo: ", logits)
            print("La: ", labels)
            print("Pr: ", probs)
            print("Loss: ", loss)
            return (loss, outputs) if return_outputs else loss
        
    train_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        do_train=True,
        evaluation_strategy="epoch",
        # logging_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        num_train_epochs=N_EPOCH,
        predict_with_generate=True,
        # warmup_steps=1000,
        warmup_ratio=WARMUP_RATIO,
        # adafactor=True,
        # seed=1,
        disable_tqdm=False,
        load_best_model_at_end=False,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        save_total_limit=3,
        report_to="wandb",
        logging_steps=100,
        # evaluation_strategy="steps",
        logging_strategy="steps",
    )


        
    trainer = Seq2SeqTrainer(
        model=model,
        args=train_args,
        train_dataset=dataset_train,
        tokenizer=tokenizer,
        eval_dataset=dataset_test,
        compute_metrics=compute_metrics,
        data_collator=smart_batching_collate_text_only,
    )

    trainer.train()

if __name__ == "__main__":
    main()