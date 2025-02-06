import textwrap
import langchain_helper as la
import streamlit as st

st.set_page_config(layout="wide")
st.title("Assistente do Youtube")

with st.sidebar:
    with st.form(key='my_form'):
        youtube_url = st.sidebar.text_area(label="Url do Vídeo", max_chars=50)
        query = st.sidebar.text_area(
            label="Me pergunte sobre algo do vídeo", max_chars=50, key="query"
        )
        submit_button = st.form_submit_button(label='Enviar')

    if query and youtube_url:
        db = la.create_vector_from_yt_url(youtube_url)
        response, docs = la.get_response_from_query(db, query)
        st.subheader("Resposta:")

        st.text(textwrap.fill(response, width=85))
