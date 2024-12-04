import pandas as pd
import openai
from openai import OpenAI

 # Replace 'file_path.pkl' with the path to your .pkl file
file_path = "C:\\Projects\\LLM_education\\translated_questions_dataset.pkl"

df = pd.read_pickle(file_path)

current_question = df.loc[(df['exam_year'] == 2018) & (df['semester'] == "A") & (df['moed'] == "A") & (df['question_number'] == "16-b") ]
msg = current_question.iloc[0]['question_translation_latex']
real_solution = current_question.iloc[0]['answer_translation_latex']


with open("api_key.txt", "r") as api_key_file:
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

msg_prefix = "Write the solution to the following question in latex format: "

reply = chat_gpt(msg_prefix + msg)
print(f"ChatGPT: {reply}")

with open("reply_16b.txt", "w") as output_file:
    output_file.write(reply)

with open("ptbs_answer_16b.txt", "w") as ptbs_file:
    ptbs_file.write(real_solution)

