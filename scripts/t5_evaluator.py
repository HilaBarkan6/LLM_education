import pandas as pd
from transformers import pipeline

# path to dataset
current_questions = pd.read_csv("./evaluation_dataset.csv")

pipe = pipeline("text2text-generation", model="google/t5_xxl_true_nli_mixture")

nli_results = []
for index, row in current_questions.iterrows():
    real_solution = row['answer_translation_latex']
    chat_solution = row['chat_answer']
    input_first = f"premise: {real_solution} hypothesis: {chat_solution}"
    input_second = f"premise: {chat_solution} hypothesis: {real_solution}"

    result1 = pipe(input_first)
    result2 = pipe(input_second)

    result = [result1, result2]
    nli_results.append(result)
    print(row['question_number'])
    print(result)

current_questions['nli_results'] = nli_results

# path to output file
current_questions.to_csv("./t5_output.csv")
