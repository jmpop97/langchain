import ollama
import time
import json
import numpy as np
from html.parser import HTMLParser
from langchain_community.embeddings import OllamaEmbeddings

def sample(filename):
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
    jsonData= {
            "start_time":start_time,
            "end_time":end_time,
            "elapsed_time":elapsed_time,
            "response":response['message']['content']
        }
    return jsonData

def modelsTest():
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
    file='prompt.txt'
    err=[]
    for model in models:
        try:
            text=sample()
            jsonData=send(model,text)
            saveJson(model,jsonData)
        except:
            err.append(model)
    saveJson("err",err)
    print(err)




class MyHTMLParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.result = []

    def handle_starttag(self, tag, attrs):
        self.result.append(f"<{tag}>")

    def handle_endtag(self, tag):
        self.result.append(f"</{tag}>")

    def handle_data(self, data):
        if data.strip():
            self.result.append(data.strip())

def split_html(html):
    parser = MyHTMLParser()
    parser.feed(html)
    return parser.result

def llm():
    chunckedSize=10
    chunckedRangeSize=2
    question="선예매 회당 몇매할 수 있어? 숫자로 알려줘."
    file="promptSample.txt"

    text=sample(file)
    testList=split_html(text)
    testList=["".join(testList[i:i+chunckedSize]) for i in range(0, len(testList), chunckedSize)]
    testListRange = ["".join(testList[i:i+chunckedRangeSize]) for i in range(len(testList)-chunckedRangeSize+1)]


    ollama_embeddings = OllamaEmbeddings(
        model="nomic-embed-text",
        # model="chatfire/bge-m3:q8_0" # BGE-M3
    )
    embedded_documents = ollama_embeddings.embed_documents(testListRange)

    embedded_query = ollama_embeddings.embed_query(question)


    # 질문(embedded_query): LangChain 에 대해서 알려주세요.
    similarity = np.array(embedded_query) @ np.array(embedded_documents).T

    # 유사도 기준 내림차순 정렬
    sorted_idx = (np.array(embedded_query) @ np.array(embedded_documents).T).argsort()[::-1]

    # 결과 출력
    print(question)
    for i, idx in enumerate(sorted_idx):
        if i<3:
            print(f"[{i}] 유사도: {similarity[idx]:.3f} | {testList[idx]}")
            print()
llm()