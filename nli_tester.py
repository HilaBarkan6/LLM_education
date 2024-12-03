print("importing...")
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from sentence_transformers import SentenceTransformer


# model 1 - nli-MiniLM2-L6-H768
print("starting nli-MiniLM2-L6-H768...")
model = AutoModelForSequenceClassification.from_pretrained('cross-encoder/nli-MiniLM2-L6-H768')
tokenizer = AutoTokenizer.from_pretrained('cross-encoder/nli-MiniLM2-L6-H768')
print("created model and tokenizer")
print("starting tokenizer...")
features = tokenizer(['A man is eating pizza', 'A black race car starts up in front of a crowd of people.'], ['A man eats something', 'A man is driving down a lonely road.'],  padding=True, truncation=True, return_tensors="pt")
print("starting eval...")
model.eval()
print("finished eval")
with torch.no_grad():
    scores = model(**features).logits
    label_mapping = ['contradiction', 'entailment', 'neutral']
    labels = [label_mapping[score_max] for score_max in scores.argmax(dim=1)]
    print(labels)


# model 2 - sentence-transformers/bert-base-nli-mean-tokens
sentences = ["This is an example sentence", "Each sentence is converted"]

print("starting bert-base-nli-mean-tokens...")
model = SentenceTransformer('sentence-transformers/bert-base-nli-mean-tokens')
print("created model")
print("starting eval...")
embeddings = model.encode(sentences)
print ("finished eval")
print(embeddings)
