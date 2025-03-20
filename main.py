from flask import Flask, request, render_template_string
from AI import send
from dotenv import load_dotenv
import os

# load .env
load_dotenv()
app = Flask(__name__)

HTML = """
<!doctype html>
<html>
<head>
    <title>Flask POST Example</title>
</head>
<body>
    <form action="/check" method="post">
        <textarea name="body" placeholder="Enter text here"></textarea>
        <button type="submit">Send Data</button>
    </form>
</body>
</html>
"""
Model=os.environ.get("model","deepseek-r1")
print(Model)
@app.route("/")
def home():
    return render_template_string(HTML)

@app.route("/check", methods=["POST"])
def check():
    body = request.form.get("body")
    print(f"Received:\n{body}")
    result=send(Model,body)
    return result["response"], 200

if __name__ == "__main__":
    app.run(debug=True)