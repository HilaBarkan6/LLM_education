import openai
from openai import OpenAI

client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key="private",
)


def chat_gpt(prompt):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

messages = [ {"role": "system", "content": 
              "You are a intelligent assistant."} ]
while True:
    message = input("User : ")
    if message:
        reply = chat_gpt(message)
        print(f"ChatGPT: {reply}")
        messages.append({"role": "assistant", "content": reply})