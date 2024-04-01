
from custom_roberta_for_sequence_classification import AveragedTopNRobertaForSequenceClassification
import json
from torch import tensor
from transformers import RobertaTokenizerFast
from rich.progress import track

model = AveragedTopNRobertaForSequenceClassification.from_pretrained(
    "best_subtask_b_mixed_trained", n=12, num_labels=6, device_map="auto")
model.eval()
tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
with open("input_data_b/data.jsonl") as file:
    sentences = []
    ids = []
    for line in file:
        text_info = json.loads(line, strict=False)
        sentences.append(text_info["text"])
        ids.append(text_info["id"])

with open("subtask_b.json", "w") as f:
    for sentence, s_id in track(zip(sentences, ids), total=len(sentences)):
        encoded = tokenizer(
            sentence, return_attention_mask=False, truncation=True, max_length=512).input_ids
        encoded = tensor([encoded]).to("cuda")
        prediction = model(encoded).logits[0].argmax().item()
        d = {"id": s_id, "label": prediction}
        json.dump(d, f)
        f.write("\n")
