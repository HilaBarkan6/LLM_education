import pandas as pd
import openai
from openai import OpenAI


# Set up chatgpt
with open(".\\api_key.txt", "r") as api_key_file:
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
file_path = ".\\evaluation_dataset\\evaluation_dataset.csv"
current_questions = pd.read_csv(file_path, engine='python')

msg_prefix = "Answer the following question. Write the solution in Latex format: "

chat_answers = []
for index, row in current_questions.iterrows():
    current_question = row['question_translation_latex']
    chat_answer = chat_gpt(msg_prefix + current_question)
    chat_answers.append(chat_answer)
    print(row['question_number'])

current_questions['chat_answer'] = chat_answers

# path to output file
current_questions.to_csv(".\\results\\output_with_solutions.csv")

