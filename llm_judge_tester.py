from openai import OpenAI

with open("api_key.txt", "r") as api_key_file:
    key = api_key_file.readline()

with open("ptbs_answer.txt", "r") as ptbs_file:
    ptbs = "".join(ptbs_file.readlines())

with open("reply_latex_format.txt", "r") as reply_file:
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
                 + "Please check if the student's solution is correct compared to the teacher's solution. Consider the data structure and the time complexity of the operations."
                 "" + "Teacher's solution: " + ptbs + "\n" + "Student's solution: " + sanity_check_1)
print(f"ChatGPT: {reply}")

# with open("llm_judge_output_1.txt", "w") as output_file:
#     output_file.write(reply)