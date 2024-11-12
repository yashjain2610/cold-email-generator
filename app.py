import streamlit as st
from langchain_community.document_loaders import WebBaseLoader

from chain import Chain
from portfolio import Portfolio
from utils import clean_text

def streamlit_app(llm,portfolio,clean_text):
    st.title("cold email generator")
    url_input = st.text_input("enter the url")
    submit_button = st.button("submit")

    if submit_button:
        loader = WebBaseLoader(url_input)
        data = clean_text(loader.load().pop().page_content)
        portfolio.load_portfolio()
        jobs = llm.extract_jobs(data)
        for job in jobs:
            skills = job.get("skills",[])
            links = portfolio.query_links(skills)
            email = llm.write_mail(job,links)
            st.code(email,language="markdown")


if __name__ == "__main__": 
    chain = Chain()
    portfolio = Portfolio()
    st.set_page_config(layout="wide")
    streamlit_app(chain,portfolio,clean_text)
