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

file_path = "C:\\Projects\\LLM_education\\results\\questions_A_to_test.csv"
df = pd.read_csv(file_path, engine='python')

current_questions = df.loc[(df['question_type'] == "A") & (df['dataset'] == "tested")]

msg_prefix = "Answer the following question. Write the solution in Latex format: "

chat_answers = []
for index, row in current_questions.iterrows():
    current_question = row['question_translation_latex']
    chat_answer = chat_gpt(msg_prefix + current_question)
    chat_answers.append(chat_answer)
    print(row['question_number'])

current_questions['chat_answer_new'] = chat_answers
current_questions.to_csv("C:\\Projects\\LLM_education\\results\\questions_A_solutions.csv")

