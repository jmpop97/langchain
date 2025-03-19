import ollama
import time


def sample():
    filename = 'prompt.txt'
    f = open(filename, 'r')     # mode = 부분은 생략해도 됨
        # 결과 사이 경계선
    lines = f.read()
    return lines
def send(model,content):
    prompt=[
        {
            'role':'user',
            'content':content
        },
    ]
    print(model)
    start_time = time.time()
    print("start - ", time.strftime('%Y.%m.%d - %H:%M:%S'))
    response = ollama.chat(model=model, messages=prompt)
    print(response['message']['content'])

    print("end - ", time.strftime('%Y.%m.%d - %H:%M:%S'))
    end_time = time.time()

    print("elapsed time - ", end_time - start_time, "seconds")

models=[
        'gemma3',
        'qwq',
        'deepseek-r1',
        'llama3.3',
        'phi4',
        'nomic-embed-text',
        'mistral',
        'qwen2.5',
        'gemma',
        'codellama',
        'neural-chat',
        'starling-lm',
        'solar'
        ]
for model in models:
    text=sample()
    send(model,text)
