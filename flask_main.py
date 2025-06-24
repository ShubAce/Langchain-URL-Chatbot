from flask import Flask, request, render_template, redirect, url_for
import os
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader
from langchain.schema import Document
import requests
from bs4 import BeautifulSoup
import validators

load_dotenv()
app = Flask(__name__)

groq_api = os.getenv("GROQ_API_KEY")

llm = ChatGroq(model="Gemma2-9b-It", api_key=groq_api)

prompt_template = """
Provide a summary of the following 
content: {text}
"""
prompt = PromptTemplate(template=prompt_template, input_variables=['text'])

def load_url_content(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        text = soup.get_text(separator="\n", strip=True)
        return [Document(page_content=text)]
    except Exception as e:
        raise RuntimeError(f"Failed to load webpage content: {e}")

@app.route("/", methods=["GET", "POST"])
def index():
    summary = None
    error = None
    if request.method == "POST":
        url = request.form.get("url")
        if not url or not url.strip():
            error = "Please enter a URL."
        elif not validators.url(url):
            error = "Invalid URL."
        else:
            try:
                if "youtube.com" in url or "youtu.be" in url:
                    loader = YoutubeLoader.from_youtube_url(youtube_url=url, add_video_info=False)
                    docs = loader.load()
                else:
                    docs = load_url_content(url)

                chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
                summary = chain.run(docs)

            except Exception as e:
                error = str(e)

    return render_template("index.html", summary=summary, error=error)

if __name__ == "__main__":
    app.run(debug=True)
