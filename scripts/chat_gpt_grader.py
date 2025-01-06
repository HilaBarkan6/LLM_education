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
        model="gpt-4o",
        temperature = 0,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

current_questions = pd.read_csv("C:\\Projects\\LLM_education\\finaldata\\finaldata.csv")
grade_results = []


for i in range(5):
    for index, row in current_questions.iterrows():
        if(row['is_chat_correct'] == "question" or row['is_ptbs_good'] == "no"):
            grade_results.append("")
        else:
            real_solution = row['answer_translation_latex']
            chat_solution = row['chat_answer']
            msg = f"You are a computer science professor at the university. \
            You will be given two solutions to a question from a test. \
            The first is the teacher solution and it is correct for sure, The second is a student solution that needs to be graded.\
            Please check the student solution. Give it a grade between 1 and 10, where 1 is for a very incorrect answer and 10 is for a great answer.\
            You are only allowed to answer a number between 1 and 10, write just the grade without any additional text.\
            \n Teacher solution: {real_solution} \n student solution: {chat_solution}"
            
            chat_grade_response = chat_gpt(msg)

            print(index)
            grade_results.append(chat_grade_response)

    current_questions[f'chatgpt_grader_10_results_iter_{i+1}'] = grade_results
    grade_results = []
    print(f"iteration {i+1} completed")

current_questions.to_csv("C:\\Projects\\LLM_education\\results\\finalgradings10.csv")