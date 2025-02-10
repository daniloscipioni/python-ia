from langchain_community.utilities import SQLDatabase
# from langchain_experimental.sql import SQLDatabaseChain
from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain.agents.agent_types import AgentType

from dotenv import load_dotenv
import os
import requests

load_dotenv()

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
db = SQLDatabase.from_uri("sqlite:///Chinook.db")
print("DB = ", db.get_context())


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
    temperature=0,
    model="gpt-4o",
    request_timeout=100000
)

# db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True)

# db_chain.run("How many albuns are there?")

agent_executor = create_sql_agent(
    llm=llm,
    toolkit=SQLDatabaseToolkit(db=db, llm=llm),
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
)

agent_executor.invoke("How many tracks are there?")
