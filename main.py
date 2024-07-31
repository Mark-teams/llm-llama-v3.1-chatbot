import os
os.system("pip install flask")
os.system("sudo apr-get install -y pciutils")
os.system("curl -fsSL https://ollama.com/install.sh | sh")

from IPython.display import clear_output
from flask import Flask, render_template,jsonify,request

app = Flask(__name__)

####
import os
import threading
import subprocess
import requests
import json

def ollama():
  os.environ['OLLAMA_HOST']='0.0.0.0:11434'
  os.environ['OLLAMA_ORIGINS']='*'
  subprocess.Popen(["ollama","serve"])

ollama_thread=threading.Thread(target=ollama)
ollama_thread.start()

from IPython.display import clear_output
os.system("ollama pull llama3.1:8b")
clear_output()

os.system("pip install -U lightrag[ollama]")

from lightrag.core.generator import Generator
from lightrag.core.component import Component
from lightrag.core.model_client import ModelClient
from lightrag.components.model_client import OllamaClient,GroqAPIClient

import time
qa_template = r"""<SYS>
you are a helpful assistant.
</Sys>
user: {{input_str}}
You:"""

class SimpleQA(Component):
  def __init__(self,model_client:ModelClient, model_kwargs:dict):
    super().__init__()
    self.generator = Generator(
        model_client=model_client,
        model_kwargs=model_kwargs,
        template=qa_template,
    )
  def call(self,input:dict)->str:
    return self.generator.call({"input_str":str(input)})
  async def acall(self,input:dict)->str:
    return await self.generator.acall({"input_str":str(input)})
   
from lightrag.components.model_client import OllamaClient
from IPython.display import Markdown, display
model={
    "model_client":OllamaClient(),
    "model_kwargs":{"model":"llama3.1:8b"}
}
qa=SimpleQA(**model)
output=qa("what is happiness")
display(Markdown(f"**Answer:** {output.data}"))

# home route that returns below text when root url is accessed
@app.route("/")
def home():
   return render_template("index.html")

@app.route("/llm_run",methods=["POST","GET"])
def llm_run():
    prompt=None
    print(prompt,request.method)
    if request.method =="POST":
       print("Value",request.form)
       prompt= request.form["search"]
       print(str(prompt))
       output=qa(prompt)
       print("running....")
    # display(Markdown(f"**Answer:** {output.data}")) str(output.data)
    return jsonify({'htmlresponse': render_template('response.html',resp=output)})
 
if __name__ == '__main__':  
   app.run()