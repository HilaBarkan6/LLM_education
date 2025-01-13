import pandas as pd
import openai
from openai import OpenAI

# Set up chatgpt
with open("C:\\Projects\\LLM_education\\api_key.txt", "r") as api_key_file:
    key = api_key_file.readline()

client = OpenAI(
    api_key=key,
)

def yes_in_result(result):
    return "yes" in result or "Yes" in result or "yes." in result or "Yes." in result

def chat_gpt(prompt):
    response = client.chat.completions.create(
        model="gpt-4o",
        temperature = 0,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

current_questions = pd.read_csv("C:\\Projects\\LLM_education\\finaldata\\finaldata.csv")

nli_results = []

for i in range(5):
    for index, row in current_questions.iterrows():
        if(row['is_chat_correct'] == "question" or row['is_ptbs_good'] == "no"):
            nli_results.append("")
        else:
            real_solution = row['answer_translation_latex']
            chat_solution = row['chat_answer']
            msg_entail = f"Does the premise entail the hypothesis? Please answer just yes or no without any additional text.\n Premise: {real_solution} \n Hypothesis: {chat_solution}"
            chat_answer_entail = chat_gpt(msg_entail)
            msg_contradict = f"Does the premise contradict the hypothesis? Please answer just yes or no without any additional text.\n Premise: {real_solution} \n Hypothesis: {chat_solution}"
            chat_answer_contradict = chat_gpt(msg_contradict)
            msg_neutral = f"Is the premise neutral to the hypothesis? Please answer just yes or no without any additional text.\n Premise: {real_solution} \n Hypothesis: {chat_solution}"
            chat_answer_neutral = chat_gpt(msg_neutral)

            msg_entail = f"Does the premise entail the hypothesis? Please answer just yes or no without any additional text.\n Premise: {chat_solution} \n Hypothesis: {real_solution}"
            chat_answer_entail_second = chat_gpt(msg_entail)
            msg_contradict = f"Does the premise contradict the hypothesis? Please answer just yes or no without any additional text.\n Premise: {chat_solution} \n Hypothesis: {real_solution}"
            chat_answer_contradict_second = chat_gpt(msg_contradict)
            msg_neutral = f"Is the premise neutral to the hypothesis? Please answer just yes or no without any additional text.\n Premise: {chat_solution} \n Hypothesis: {real_solution}"
            chat_answer_neutral_second = chat_gpt(msg_neutral)


            result_entail = [chat_answer_entail, chat_answer_entail_second]
            result_contradict = [chat_answer_contradict, chat_answer_contradict_second]
            result_neutral = [chat_answer_neutral, chat_answer_neutral_second]
            result = ""

            if yes_in_result(result_entail):
                if not yes_in_result(result_contradict):
                    result = "entail"
                else:
                    result = "inconclusive"
            if yes_in_result(result_contradict):
                if not yes_in_result(result_entail):
                    result = "contradict"
                else:
                    result = "inconclusive"
            if not yes_in_result(result_entail) and not yes_in_result(result_contradict) and yes_in_result(result_neutral):
                result = "neutral"
            if result == "":
                result = f"entail: {result_entail} contradict: {result_contradict} neutral: {result_neutral}"

            print(index, result)
            print(result_entail, result_contradict, result_neutral)

            nli_results.append(result)

    current_questions[f'nli_results_iter_{i+1}'] = nli_results
    nli_results = []

current_questions.to_csv("C:\\Projects\\LLM_education\\results\\chatgpt_as_nli_output_final.csv")