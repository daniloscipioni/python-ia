from langchain_openai.embeddings.base import OpenAIEmbeddings
from langchain_community.document_loaders.youtube import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import requests
import os

load_dotenv()
flow_api_url = os.getenv("FLOW_API_URL")
flow_api_url_token = os.getenv("FLOW_API_URL_TOKEN")
flow_api_client_id = os.getenv("CLIENT_ID")
flow_api_client_secret = os.getenv("CLIENT_SECRET")
flow_api_tenant = os.getenv("FLOW_TENANT")
flow_api_agent = os.getenv("FLOW_AGENT")
flow_apps = os.getenv("APPS")
openai_api_key = os.getenv("OPENAI_API_KEY")

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
token = getToken()["access_token"]

embeddings = OpenAIEmbeddings(
    default_headers={
        "FlowTenant": flow_api_tenant,
        "FlowAgent": flow_api_agent,
        "Authorization": f"Bearer {token}",
    },
    base_url=flow_api_url
)


def create_vector_from_yt_url(video_url: str) -> FAISS:
    loader = YoutubeLoader.from_youtube_url(video_url, language="pt")
    transcript = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(transcript)

    db = FAISS.from_documents(docs, embeddings)
    return db


def get_response_from_query(db, query, k=4):
    docs = db.similarity_search(query, k=k)
    docs_page_content = " ".join([d.page_content for d in docs])

    token = getToken()["access_token"]
    expires_in = getToken()["expires_in"]
    try:
        llm = ChatOpenAI(
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
            request_timeout=100000
        )

        template = ChatPromptTemplate([
            ("user", """Você é um assistente que responde perguntas sobre vídeos do youtube baseado
        na transcrição do vídeo.

        Responda a seguinte pergunta: {pergunta}
        Procurando nas seguintes transcrições: {docs}

        Use somente informação da transcrição para responder a pergunta. Se você não sabe, responda
        com "Eu não sei".

        Suas respostas devem ser bem detalhadas e verbosas."""),
        ])

        # chain = LLMChain(llm=llm, prompt=template, output_key="answer")
        prompt_value = template.invoke(
            {"pergunta": query, "docs": docs_page_content}
        )

        # response = chain({"pergunta": query, "docs": docs_page_content})

        response = llm.invoke(prompt_value).content

        return response, docs

    except Exception as e:
        print("Ocorreu um erro", e)


if __name__ == "__main__":
    db = create_vector_from_yt_url("https://www.youtube.com/watch?v=VwkGdSHmoJ4")
    response, docs = get_response_from_query(
        db, "O que é falado sobre o assunto?"
    )

    print(response)
