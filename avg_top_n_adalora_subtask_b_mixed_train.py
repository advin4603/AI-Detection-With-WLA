from classification_dataset import MachineClassificationDatasetMixed, MachineClassificationDataset
from transformers import TrainingArguments, Trainer
from sklearn.metrics import accuracy_score
import numpy as np
from peft import get_peft_model, AdaLoraConfig, TaskType
from transformers import EarlyStoppingCallback
from custom_roberta_for_sequence_classification import AveragedTopNRobertaForSequenceClassification
import gc

print("loading datasets")
main_dataset = MachineClassificationDatasetMixed(
    "SemEval2024-Task8/SubtaskB/subtaskB_train.jsonl", "SemEval2024-Task8/SubtaskB/subtaskB_dev.jsonl", binary=False)
train_dataset, eval_dataset = main_dataset.split(80, 20)
del main_dataset
gc.collect()  # delete heavy main_dataset
print("loaded train dataset")


def compute_metrics(p):
    pred, labels = p
    pred = np.argmax(pred, axis=1)

    accuracy = accuracy_score(y_true=labels, y_pred=pred)

    return {"accuracy": accuracy}


learning_rate = 5e-4
per_device_train_batch_size = 8
weight_decay = 5e-5
warmup_ratio = 0.01


args = TrainingArguments(learning_rate=learning_rate, per_device_train_batch_size=per_device_train_batch_size, weight_decay=weight_decay, metric_for_best_model="eval_accuracy", greater_is_better=True, output_dir="/scratch/ayan.datta/avg_top_n_subtask_b_mixed", evaluation_strategy="steps",
                         eval_steps=500, save_strategy="steps", save_total_limit=2, load_best_model_at_end=True, disable_tqdm=False)


init_r = 12
target_r = 8
lora_alpha = 200
lora_dropout = 0.4

peft_config = AdaLoraConfig(task_type=TaskType.SEQ_CLS,
                            inference_mode=False, init_r=init_r, target_r=target_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, modules_to_save=["layer_weights"])

n = 12
model = AveragedTopNRobertaForSequenceClassification.from_pretrained(
    "roberta-base", num_labels=6, n=n)
model = get_peft_model(model, peft_config=peft_config)


trainer = Trainer(model=model, args=args, train_dataset=train_dataset, eval_dataset=eval_dataset,
                  compute_metrics=compute_metrics, data_collator=train_dataset.collate, callbacks=[EarlyStoppingCallback(early_stopping_patience=5, early_stopping_threshold=0.001)])


print("training")
trainer.train()


trainer.save_model(output_dir="best_subtask_b_mixed_trained")


print("loading eval dataset")
official_eval_dataset = MachineClassificationDataset(
    "SemEval2024-Task8/SubtaskB/subtaskB_dev.jsonl")
print("loaded eval dataset")

print(trainer.evaluate(eval_dataset=official_eval_dataset))
