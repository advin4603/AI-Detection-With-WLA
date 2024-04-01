from classification_dataset import MachineClassificationDataset, MachineClassificationDatasetMixed
from custom_roberta_for_sequence_classification import AveragedTopNRobertaForSequenceClassification
from torch import tensor, no_grad
from transformers import RobertaTokenizerFast
from rich.progress import track
from transformers import TrainingArguments, Trainer, set_seed
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import numpy as np
import gc

model = AveragedTopNRobertaForSequenceClassification.from_pretrained(
    "best_subtask_a_mixed_trained", n=12, device_map="auto")
model.eval()
tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")

print("loading test dataset")
official_test_dataset = MachineClassificationDataset(
    "LabelledTestData/subtaskA_monolingual.jsonl")
print("loaded test dataset")

learning_rate = 5e-4
per_device_train_batch_size = 10
weight_decay = 5e-5
warmup_ratio = 0.1


def compute_metrics(p):
    pred, labels = p
    pred = np.argmax(pred, axis=1)

    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    recall = recall_score(y_true=labels, y_pred=pred)
    precision = precision_score(y_true=labels, y_pred=pred)
    f1 = f1_score(y_true=labels, y_pred=pred)

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


args = TrainingArguments(learning_rate=learning_rate, per_device_train_batch_size=per_device_train_batch_size, per_device_eval_batch_size=per_device_train_batch_size, weight_decay=weight_decay, metric_for_best_model="eval_accuracy", greater_is_better=True, output_dir="/scratch/ayan.datta/avg_top_n_subtask_a_full_mixed", evaluation_strategy="steps",
                         eval_steps=500, save_strategy="steps", save_total_limit=2, load_best_model_at_end=True, disable_tqdm=False)


set_seed(42)


trainer = Trainer(model=model, args=args, eval_dataset=official_test_dataset,
                  compute_metrics=compute_metrics, data_collator=official_test_dataset.collate)


print(trainer.evaluate(eval_dataset=official_test_dataset))

print("loading eval dataset")
official_eval_dataset = MachineClassificationDataset(
    "SemEval2024-Task8/SubtaskA/subtaskA_dev_monolingual.jsonl")
print("loaded eval dataset")

print(trainer.evaluate(eval_dataset=official_eval_dataset))

print("loading datasets")
main_dataset = MachineClassificationDatasetMixed(
    "SemEval2024-Task8/SubtaskA/subtaskA_train_monolingual.jsonl", "SemEval2024-Task8/SubtaskA/subtaskA_dev_monolingual.jsonl")
train_dataset, eval_dataset = main_dataset.split(80, 20)
del main_dataset
del train_dataset
gc.collect()  # delete heavy main_dataset
print("loaded eval dataset")


print(trainer.evaluate(eval_dataset=eval_dataset))
