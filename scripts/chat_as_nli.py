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

df = pd.read_csv("C:\\Projects\\LLM_education\\results\\questions_A_with_chat_solutions_checked.csv")
current_questions = df.loc[(df['is_chat_correct'] != "question") & (df['is_ptbs_good'] != "no")]

nli_results_entail = []
nli_results_contradict = []
nli_results_neutral = []

for index, row in current_questions.iterrows():
    real_solution = row['answer_translation_latex']
    chat_solution = row['chat_answer_new']
    msg_entail = f"Does the premise entail the hypothesis? Please answer yes or no.\n Premise: {real_solution} \n Hypothesis: {chat_solution}"
    chat_answer_entail = chat_gpt(msg_entail)
    msg_contradict = f"Does the premise contradict the hypothesis? Please answer yes or no.\n Premise: {real_solution} \n Hypothesis: {chat_solution}"
    chat_answer_contradict = chat_gpt(msg_contradict)
    msg_neutral = f"Is the premise neutral to the hypothesis? Please answer yes or no.\n Premise: {real_solution} \n Hypothesis: {chat_solution}"
    chat_answer_neutral = chat_gpt(msg_neutral)

    msg_entail = f"Does the premise entail the hypothesis? Please answer yes or no.\n Premise: {chat_solution} \n Hypothesis: {real_solution}"
    chat_answer_entail_second = chat_gpt(msg_entail)
    msg_contradict = f"Does the premise contradict the hypothesis? Please answer yes or no.\n Premise: {chat_solution} \n Hypothesis: {real_solution}"
    chat_answer_contradict_second = chat_gpt(msg_contradict)
    msg_neutral = f"Is the premise neutral to the hypothesis? Please answer yes or no.\n Premise: {chat_solution} \n Hypothesis: {real_solution}"
    chat_answer_neutral_second = chat_gpt(msg_neutral)


    result_entail = [chat_answer_entail, chat_answer_entail_second]
    result_contradict = [chat_answer_contradict, chat_answer_contradict_second]
    result_neutral = [chat_answer_neutral, chat_answer_neutral_second]

    print(row['question_number'])
    print(result_entail)
    print(result_contradict)
    print(result_neutral)
    print("\n")

    nli_results_entail.append(result_entail)
    nli_results_contradict.append(result_contradict)
    nli_results_neutral.append(result_neutral)


current_questions['nli_results_entail'] = nli_results_entail
current_questions['nli_results_contradict'] = nli_results_contradict
current_questions['nli_results_neutral'] = nli_results_neutral

current_questions.to_csv("C:\\Projects\\LLM_education\\results\\questionsA_chatgpt_as_nli_output.csv")