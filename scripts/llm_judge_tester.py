from openai import OpenAI

with open("../api_key.txt", "r") as api_key_file:
    key = api_key_file.readline()

with open("../results/ptbs_answer_16b_updated.txt", "r") as ptbs_file:
    ptbs = "".join(ptbs_file.readlines())

with open("../results/reply_16b.txt", "r") as reply_file:
    reply_latex = "".join(reply_file.readlines())

client = OpenAI(
    api_key=key,
)

def chat_gpt(prompt):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

sanity_check_1 = "let's use a sorted array. Insert wil take O(logn), search will take O(logn), delete will take O(logn)"


reply = chat_gpt("Hi chat, you are a professor at a fancy university grading a data structure test. The first solution below is the official solution, the second is a student's solution."
                 + "The teacher's solution is correct. Please check if the student's solution is similar compared to the teacher's solution. Consider the data structure and the time complexity of the operations. After comparing the answers step by step, explicitly say if the student is correct or not: "
                 "" + "Teacher's solution: " + ptbs + "\n" + "Student's solution: " + reply_latex)
print(f"ChatGPT: {reply}")

# with open("llm_judge_output_16_b_similar.txt", "w") as output_file:
#     output_file.write(reply)