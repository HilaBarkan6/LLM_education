print("importing...")
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from sentence_transformers import SentenceTransformer

with open("reply_latex_format.txt", "r") as reply_file:
    chat_gpt_reply = "".join(reply_file.readlines())

official_solution = "\\textbf{Central Data Structure:} AVL Tree. \\textbf{Brief Description:} We will use the standard operations of an AVL tree. \\textbf{Complexity Explanation:} All operations are in worst-case in \(O(\log n)\), and therefore, in particular, amortized directly from the definition of amortized."

# model 1 - nli-MiniLM2-L6-H768
print("starting nli-MiniLM2-L6-H768...")
model = AutoModelForSequenceClassification.from_pretrained('cross-encoder/nli-MiniLM2-L6-H768')
tokenizer = AutoTokenizer.from_pretrained('cross-encoder/nli-MiniLM2-L6-H768')
print("created model and tokenizer")
print("starting tokenizer...")
features = tokenizer([chat_gpt_reply, official_solution], [official_solution, chat_gpt_reply],  padding=True, truncation=True, return_tensors="pt")
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
