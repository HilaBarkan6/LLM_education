print("importing...")
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

print("starting nli-MiniLM2-L6-H768...")
model = AutoModelForSequenceClassification.from_pretrained('cross-encoder/nli-MiniLM2-L6-H768')
tokenizer = AutoTokenizer.from_pretrained('cross-encoder/nli-MiniLM2-L6-H768')
print("created model and tokenizer")

current_questions = pd.read_csv("~/Projects/LLM_education/results/2018_AB_chat_solutions.csv")

nli_results = []
for index, row in current_questions.iterrows():
    real_solution = row['answer_translation_latex']
    chat_solution = row['chat_answer']

    print("starting tokenizer...")
    features = tokenizer([real_solution, chat_solution], [chat_solution, real_solution], padding=True, truncation=True,
                         return_tensors="pt")
    print("starting eval...")
    model.eval()
    print("finished eval")

    with torch.no_grad():
        scores = model(**features).logits
        label_mapping = ['contradiction', 'entailment', 'neutral']
        labels = [label_mapping[score_max] for score_max in scores.argmax(dim=1)]

    nli_output = labels
    nli_results.append(labels)
    print(row['question_number'])

current_questions['nli'] = nli_results
current_questions.to_csv("~/Projects/LLM_education/results/2018_AB_nli.csv")
