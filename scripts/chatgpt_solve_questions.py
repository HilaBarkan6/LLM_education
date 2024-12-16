import pandas as pd
import openai
from openai import OpenAI

# Set up chatgpt
with open("C:\\Projects\\LLM_education\\api_key.txt", "r") as api_key_file:
    key = api_key_file.readline()

client = OpenAI(
    api_key=key,
)


def chat_gpt(prompt):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()



# Set up data

file_path = "C:\\Projects\\LLM_education\\translated_questions_dataset.pkl"
df = pd.read_pickle(file_path)

current_questions = df.loc[(df['dataset'] == "tested") & (df['has_solution'] == True)]
current_questions = current_questions.loc[(df['question_type'] == 'b') | (df['question_type'] == 'B') | (df['question_type'] == 'c') | (df['question_type'] == 'C')]
current_questions = current_questions.loc[(df['answer_translation_latex'] != "na") & (df['question_translation_latex'] != "na") & (df['answer_translation_latex'] != "")]

msg_prefix = "Write the solution to the following question in latex format: "

chat_answers = []
for index, row in current_questions.iterrows():
    current_question = row['question_translation_latex']
    chat_answer = chat_gpt(msg_prefix + current_question)
    chat_answers.append(chat_answer)
    print(row['question_number'])

current_questions['chat_answer'] = chat_answers
current_questions.to_csv("C:\\Projects\\LLM_education\\results\\big_data_chat_solutions.csv")

