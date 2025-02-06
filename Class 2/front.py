import langchain_helper as la
import streamlit as st

st.set_page_config(layout="wide")
st.title("Gerador de nomes de empresas")
segmento = st.sidebar.text_area(label="Qual é o segmento da sua empresa")

if segmento:
    response = la.generate_company_name(segmento=segmento)

    st.text(response)
