from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import requests
from langchain.schema.output_parser import StrOutputParser
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import os
from langchain.schema.runnable import RunnablePassthrough,RunnableLambda
from langchain.utilities import DuckDuckGoSearchAPIWrapper
from langchain_google_genai import ChatGoogleGenerativeAI

import json

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



Search_prompt=ChatPromptTemplate.from_messages(
    [
        (
            "user",
            "Write 3 google search queries to searh online that form an"
            "objective opinion from the following :{question}\n"
            "You must respond with a list of strings in the following format:"
            '["query1","query2","query3"]',
        ),
    ]
)


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



search_question_chain=Search_prompt|ChatGoogleGenerativeAI(model="gemini-pro")|StrOutputParser()|json.loads

# search_question_chain.invoke(
#     {
#         "question":"What is difference between langsmith and langchain?",
#     }
# )
scrape_and_summarize_chain= RunnablePassthrough.assign(
summary=RunnablePassthrough.assign(
    text=lambda x:scrape_text(x["url"])[:10000]
) |Summary_prompt|ChatGoogleGenerativeAI(model="gemini-pro")|StrOutputParser()
)|(lambda x:f"URL:{x['url']}\n\nSummary:{x['summary']}")


web_search_chain=RunnablePassthrough.assign(
    urls=lambda x :web_search(x["question"])
    )| (lambda x:[{"question":x["question"],"url":u} for u in x["urls"]]) | scrape_and_summarize_chain.map()

full_research_chain=search_question_chain|( lambda x:[{"question":q} for q in x])|web_search_chain.map()


WRITER_SYSTEM_PROMPT = "You are an AI critical thinker research assistant. Your sole purpose is to write well written, critically acclaimed, objective and structured reports on given text."  # noqa: E501

RESEARCH_REPORT_TEMPLATE = """Information:
--------
{research_summary}
--------
Using the above information, answer the following question or topic: "{question}" in a detailed report -- \
The report should focus on the answer to the question, should be well structured, informative, \
in depth, with facts and numbers if available and a minimum of 1,200 words.
You should strive to write the report as long as you can using all relevant and necessary information provided.
You must write the report with markdown syntax.
You MUST determine your own concrete and valid opinion based on the given information. Do NOT deter to general and meaningless conclusions.
Write all used source urls at the end of the report, and make sure to not add duplicated sources, but only one reference for each.
You must write the report in apa format.
Please do your best, this is very important to my career."""  

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", WRITER_SYSTEM_PROMPT),
        ("user", RESEARCH_REPORT_TEMPLATE),
    ]
)


def collapse_list_of_lists(list_of_lists):
    content=[]
    for l in list_of_lists:
        content.append("\n\n".join(l))
    return "\n\n".join(content)

chain=RunnablePassthrough.assign(
    research_summary=full_research_chain|collapse_list_of_lists
)|prompt|ChatGoogleGenerativeAI(model="gemini-pro",convert_system_message_to_human=True)|StrOutputParser()

output=chain.invoke(
    {
        "question":"Bubble Diagram generation for architectural floor plans using neural networks",
    }
)


