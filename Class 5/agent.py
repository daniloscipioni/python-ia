from langchain_community.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from langchain_openai import ChatOpenAI

from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain.agents.agent_types import AgentType

from langchain.chains import LLMMathChain
from langchain_community.utilities.serpapi import SerpAPIWrapper
from langchain.agents import initialize_agent, Tool
from langchain.prompts import MessagesPlaceholder
from langchain.memory import ConversationBufferMemory

from dotenv import load_dotenv
import os
import requests

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
openai_organization = os.getenv("OPENAI_ORGANIZATION")

db = SQLDatabase.from_uri("sqlite:///Chinook.db")
print("DB = ", db.get_usable_table_names())

openai_api_key = os.getenv("OPENAI_API_KEY")
openai_organization = os.getenv("OPENAI_ORGANIZATION")
flow_api_url = os.getenv("FLOW_API_URL")
flow_api_url_token = os.getenv("FLOW_API_URL_TOKEN")
flow_api_client_id = os.getenv("CLIENT_ID")
flow_api_client_secret = os.getenv("CLIENT_SECRET")
flow_api_tenant = os.getenv("FLOW_TENANT")
flow_api_agent = os.getenv("FLOW_AGENT")
flow_apps = os.getenv("APPS")
openai_api_key = os.getenv("OPENAI_API_KEY")

def getToken() -> object:

    payload = {
        "clientId": flow_api_client_id,
        "clientSecret": flow_api_client_secret,
        "appToAccess": flow_apps
    }

    headers = {
        'Content-Type': 'application/json',
        'accept': '*/*',
        'FlowAgent': flow_api_agent,
        'FlowTenant': flow_api_tenant
    }

    # Fazendo a requisição POST
    response = requests.post(flow_api_url_token, json=payload, headers=headers)

    # Verificando o status da resposta
    if response.status_code == 200:
        return response.json()
    else:
        print('Falha na requisição:', response.status_code)
        print('Mensagem:', response.text)

token = getToken()["access_token"]

llm = ChatOpenAI(
    openai_api_base=flow_api_url,
    default_headers={
        "FlowTenant": flow_api_tenant,
        "FlowAgent": flow_api_agent,
        "Authorization": f"Bearer {token}",
    },
    max_retries=2,
    n=1,
    temperature=0.4,
    model="gpt-4o-mini",
    request_timeout=100000
)

db_chain = SQLDatabaseChain.from_llm(llm, db)



llm_math_chain = LLMMathChain.from_llm(llm=llm, verbose=True)
search = SerpAPIWrapper()

tools = [
    Tool(
        name="FooBar-DB",
        func=db_chain.run,
        description="useful for when you need to answer questions about FooBar. Input should be in the form of a question containing full context",
    ),
    Tool(
        name="Search",
        func=search.run,
        description="useful for when you need to answer questions about current events. You should ask targeted questions",
    ),
    Tool(
       name="Calculator",
       func=llm_math_chain.run,
       description="useful for when you need to answer questions about math",
    )
]

memory = ConversationBufferMemory(memory_key="memory", return_messages=True)

agent_kwargs = {
    "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
}

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True,
    agent_kwargs=agent_kwargs,
    memory=memory,
)


def handle_chat(query):
    response = agent.invoke(query)
    return response
#while True:
#    user_input = input("Enter your query or type 'exit' to quit: ")

#    if user_input.lower() == "exit":
#        break

#    response = agent.invoke(user_input)

#    print(response)