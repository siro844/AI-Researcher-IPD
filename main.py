from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import requests
from langchain.schema.output_parser import StrOutputParser
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import os
from langchain.schema.runnable import RunnablePassthrough,RunnableLambda
from langchain.utilities import DuckDuckGoSearchAPIWrapper

load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")

RESULTS_PER_QUESTION=3

ddg_search=DuckDuckGoSearchAPIWrapper()

def web_search(query:str,num_results:int=RESULTS_PER_QUESTION):
    results=ddg_search.results(query,num_results)
    return [r["link"] for r in results]



SUMMARY_TEMPLATE="""{text}
--------------------

Using the above text, answer in short the following question:
>{question}
--------------------
"""




template="""
{text}
Using the above text, answer the following question:
>{question}
If the question cannot be answered by the text ,simply summarize the text.Include all factual information,numbers,stats etc.
"""


Summary_prompt=ChatPromptTemplate.from_template(
    template=SUMMARY_TEMPLATE
)
prompt=ChatPromptTemplate.from_template(
    template=template,
    
)
url="https://blog.langchain.dev/announcing-langsmith/"

def scrape_text(website:str):
    try:
        response=requests.get(website)

        if response.status_code==200:
            soup=BeautifulSoup(response.text,'html.parser')

            page_text=soup.get_text(separator="",strip=True)

            return page_text
        else:
            return f"Failed with status:{response.status_code}"
    except Exception as e:
        print(e)
        return "Failed with exception:{e}"

scrape_and_summarize_chain= RunnablePassthrough.assign(
    text=lambda x:scrape_text(x["url"])[:10000]
) |Summary_prompt|ChatOpenAI(model='gpt-3.5-turbo-1106')|StrOutputParser()


chain=RunnablePassthrough.assign(
    urls=lambda x :web_search(x["question"])
    )| (lambda x:[{"question":x["question"],"url":u} for u in x["urls"]]) | scrape_and_summarize_chain.map()


output=chain.invoke(
    {
        "question":"what is langsmith",
    }
)


