import pandas as pd
import openai
from openai import OpenAI

# Set up chatgpt
with open("../api_key.txt", "r") as api_key_file:
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

# file_path = "C:\\Projects\\LLM_education\\translated_questions_dataset.pkl"
# df = pd.read_pickle(file_path)

# current_questions = df.loc[(df['exam_year'] == 2018) & (df['semester'] == "A") & (df['moed'] == "B") & (df['dataset'] == "tested")]

# msg_prefix = "Write the solution to the following question in latex format: "

# chat_answers = []
# for index, row in current_questions.iterrows():
#     current_question = row['question_translation_latex']
#     chat_answer = chat_gpt(msg_prefix + current_question)
#     chat_answers.append(chat_answer)
#     print(row['question_number'])

# current_questions['chat_answer'] = chat_answers
# current_questions.to_csv("C:\\Projects\\LLM_education\\2018_AB_chat_solutions.csv")

current_questions = pd.read_csv("C:\\Projects\\LLM_education\\results\\2018_AB_chat_solutions.csv")

llm_judge_results = []
for index, row in current_questions.iterrows():
    real_solution = row['answer_translation_latex']
    chat_solution = row['chat_answer']
    llm_as_a_judge_output = chat_gpt("You are a professor at a university grading a data structures test. \
                                     The first solution below is the offical solution, the second is a student's solution.\
                                     The offical solution is correct. Please check if the student's solution is similar compared to the official solution.\
                                     Consider the data structure and time complexity of the operations. \
                                     After comparing the solution step by step, explicitly say if the student is correct or not. \
                                     Offical solution: " + real_solution + "Student solution: " + chat_solution)
    llm_judge_results.append(llm_as_a_judge_output)
    print(row['question_number'])

current_questions['judge'] = llm_judge_results
current_questions.to_csv("C:\\Projects\\LLM_education\\results\\2018_AB_llm_judge.csv")

