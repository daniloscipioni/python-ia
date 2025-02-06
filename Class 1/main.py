from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

import requests

from dotenv import load_dotenv
import os

load_dotenv()
flow_api_url = os.getenv("FLOW_API_URL")
flow_api_url_token = os.getenv("FLOW_API_URL_TOKEN")
flow_api_client_id = os.getenv("CLIENT_ID")
flow_api_client_secret = os.getenv("CLIENT_SECRET")
flow_api_tenant = os.getenv("FLOW_TENANT")
flow_api_agent = os.getenv("FLOW_AGENT")
flow_apps = os.getenv("APPS")


# FLOW_API_URL_TOKEN=https://flow.ciandt.com/auth-engine-api/v1/api-key/token
# FLOW_API_URL=https://flow.ciandt.com/ai-orchestration-api/v1/openai
# CLIENT_ID=**********
# CLIENT_SECRET=**********
# FLOW_TENANT=**********
# FLOW_AGENT=**********
# APPS=**********


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

# MODELS ALLOWED  'gpt-4-32k', 'gpt-4-0314', 'gpt-4-32k-0314', 'gpt-4o', 'gpt-4o-mini', 'gpt-35-turbo', 'gpt-35-turbo-instruct', 'gpt-35-turbo-16k', 'text-embedding-ada-002', 'o1-mini', 'o1-preview'


def generate_company_name():
    token = getToken()["access_token"]
    expires_in = getToken()["expires_in"]

    flow = ChatOpenAI(
        openai_api_base=flow_api_url,
        default_headers={
            "FlowTenant": flow_api_tenant,
            "FlowAgent": flow_api_agent,
            "Authorization": f"Bearer {token}",
        },
        max_retries=2,
        n=1,
        temperature=1,
        model="gpt-4o",
        request_timeout=100000,
    )

    chat_template = [
        SystemMessage(content="Você é um assistente IA que sempre responde em português"),
        HumanMessage(
            content="Me de 5 sugestoes de nomes de empresas do segmento Pets que sejam criativas!")
    ]

    return flow.invoke(chat_template).content


if __name__ == "__main__":
    print(generate_company_name())
