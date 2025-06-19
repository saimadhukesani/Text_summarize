import streamlit as st
import validators
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader

# Streamlit page config
st.set_page_config(page_title="Langchain: Summarize YT or Website")
st.title("Langchain: Summarize YT or Website Content")
st.subheader("Paste any YouTube or Website URL below to get a summary")

# Sidebar for API key input
# with st.sidebar:
#     groq_api_key = st.text_input("Enter your Groq API Key:", value="", type="password")
import os
groq_api_key=os.getenv("Groq_api")

# URL input
url = st.text_input("Paste YouTube or Website URL", label_visibility="collapsed")

# Prompt template
prompt_template = """
Provide a summary of the following content in 300 words:
Content: {text}
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

# Button click event
if st.button("Summarize the content from YT or Website"):
    if not groq_api_key.strip() or not url.strip():
        st.error(" Please provide both a Groq API key and a valid URL.")
    elif not validators.url(url):
        st.error("Please enter a valid URL (YouTube or Website).")
    else:
        try:
            with st.spinner("Loading and summarizing content..."):

                # Initialize LLM only after inputs are validated
                llm = ChatGroq(groq_api_key=groq_api_key, model="llama3-8b-8192")

                # Load documents from YouTube or Web
                if "youtube.com" in url or "youtu.be" in url:
                    loader = YoutubeLoader.from_youtube_url(url, add_video_info=False)
                else:
                    loader = UnstructuredURLLoader(
                        urls=[url],
                        ssl_verify=True,
                        headers={
                            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"
                        }
                    )

                docs = loader.load()

                if not docs:
                    st.error(" Failed to load content from the given URL.")
                else:
                    # Summarization chain
                    chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
                    output_summary = chain.run(docs)
                    st.success("Summary generated:")
                    st.write(output_summary)

        except Exception as E:
            st.error(" An unexpected error occurred:")
            st.exception(E)
