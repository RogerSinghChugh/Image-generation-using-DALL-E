import os
import openai
import gradio as gr
from dotenv import load_dotenv
import requests
from PIL import Image
import numpy as np
from io import BytesIO

load_dotenv()

openai.api_key = os.getenv("open_ai_token") #Personal

start_sequence = "\nAI:"
restart_sequence = "\nHuman: "

prompt = "The following is a conversation with an AI assistant. The assistant is helpful, creative, clever, and very friendly.\n\nHuman: Hello, who are you?\nAI: I am an AI created by OpenAI. How can I help you today?\nHuman: "

#### Davinci-003 - GPT 3 model + DALL E
def openai_create(prompt):

    response1 = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a helpful assistant that converses with the user."},
        {"role": "user", "content": f"Make a DALL E prompt under 400 chars for the prompt: {prompt}"}
    ],
    temperature=0,
    max_tokens=2000,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0.6,
    stop=[" Human:", " AI:"]
    )
    response2 = openai.Image.create(
        prompt=response1['choices'][0]['message']['content'],
        n=1,
        size="1024x1024"
    )
    image_url = response2['data'][0]['url']

    print(image_url)
    print(response1['choices'][0]['message']['content'])
    image_data = requests.get(image_url).content
    image = Image.open(BytesIO(image_data))

    # return response.choices[0].text
    return np.array(image)

def chatgpt_clone(input, history):
    history = history or []
    s = list(sum(history, ()))
    s.append(input)
    inp = ' '.join(s)
    output = openai_create(inp)
    print(f"Bot: {output}")
    history.append((input, output))
    return (output, None)


iface = gr.Interface(
    fn=openai_create,
    inputs="text",
    outputs="image",
    title="DALL·E Image Generation",
    description="Enter a prompt and see DALL·E generate an image.",
    theme="default"
)

iface.launch(debug = True) # Development
# demo.launch(server_name="0.0.0.0", share=False) # Production
