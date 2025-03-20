import ollama
import time
import json


def sample():
    filename = 'prompt.txt'
    f = open(filename, 'r')     # mode = 부분은 생략해도 됨
        # 결과 사이 경계선
    lines = f.read()
    return lines

def saveJson(name,jsondata):
    # json 파일로 저장
    with open(f'./results/{name}.json', 'w',encoding='UTF-8') as f : 
        json.dump(jsondata, f, indent=4,ensure_ascii=False)

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
    elapsed_time=end_time - start_time
    print("elapsed time - ", elapsed_time, "seconds")
    jsonData=        {
            "start_time":start_time,
            "end_time":end_time,
            "elapsed_time":elapsed_time,
            "response":response['message']['content']
        }
    return jsonData


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
err=[]
for model in models:
    try:
        text=sample()
        jsonData=send(model,text)
        saveJson(model,jsonData)
    except:
        err.append(model)
print(err)