import pandas as pd
import openai
from openai import OpenAI

 # Replace 'file_path.pkl' with the path to your .pkl file
file_path = "C:\\Projects\\LLM_education\\translated_questions_dataset.pkl"

df = pd.read_pickle(file_path)

q16a_2018_aa = df.loc[(df['exam_year'] == 2018) & (df['semester'] == "A") & (df['moed'] == "A") & (df['question_number'] == "16-a") ]
msg = q16a_2018_aa.iloc[0]['question_translation_latex']
real_solution = q16a_2018_aa.iloc[0]['answer_translation_latex']


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

reply = chat_gpt(msg)
print(f"ChatGPT: {reply}")

with open("reply.txt", "w") as output_file:
    output_file.write(reply)

with open("ptbs answer", "w") as ptbs_file:
    ptbs_file.write(real_solution)

