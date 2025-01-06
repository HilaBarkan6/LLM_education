import pandas as pd
import openai
from openai import OpenAI


# Set up chatgpt
with open("/Users/stavfn/Projects/LLM_education/api_key.txt", "r") as api_key_file:
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

file_path = "~/Projects/LLM_education/results/big_data_chat_solutions_checked.csv"
df = pd.read_csv(file_path)

current_questions = df.loc[df['is_chat_correct'] == "question"]

msg_prefix = "Answer the following question. Write only the solution without the question: "

chat_answers = []
for index, row in current_questions.iterrows():
    current_question = row['question_translation_latex']
    chat_answer = chat_gpt(msg_prefix + current_question)
    chat_answers.append(chat_answer)
    print(row['question_number'])

current_questions['chat_answer_new'] = chat_answers
current_questions.to_csv("~/Projects/LLM_education/results/questions_no_solution_solutions.csv")

