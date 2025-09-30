import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

DB_FAISS_PATH="vectorstore/db_faiss"
@st.cache_resource
def get_vectorstore():
    embedding_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db=FAISS.load_local(DB_FAISS_PATH,embedding_model,allow_dangerous_deserialization=True)
    return db


def set_custom_prompt(custom_prompt_template):
    prompt=PromptTemplate(template=custom_prompt_template,input_variables=["context","question"])
    return prompt


def load_llm():
    load_dotenv()
    GROQ_API_KEY=os.environ.get("GROQ_API_KEY")
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=GROQ_API_KEY,      # ✅ pass API key here
        temperature=0.5,
        # max_output_tokens=512       # ✅ Groq uses max_output_tokens, not max_length
    )
    return llm


def main():
    st.title("Ask Chabot")

    

    if 'messages' not in st.session_state:
        st.session_state.messages=[]

    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    prompt=st.chat_input("Pass your prompt")

    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role':'user','content':'prompt'})

        custom_prompt_template="""
            Use the pieces of information provided in the context to answer user's question.
            If you dont know the answer, just say that you dont know, dont try to make up an answer. 
            Dont provide anything out of the given context

            Context: {context}
            Question: {question}

            Start the answer directly. No small talk please."""

        

        try:
            vectorstore=get_vectorstore()
            if vectorstore is None:
                st.error("Failed to load vectorstore")

            qa_chain = RetrievalQA.from_chain_type(
                llm=load_llm(),   # your LLM loader (e.g., ChatGoogleGenerativeAI)
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
                return_source_documents=True,   # typo fixed
                chain_type_kwargs={
                    "prompt": set_custom_prompt(custom_prompt_template)  # custom prompt
                }
            )
            

            response=qa_chain.invoke({'query':prompt})

            result=response['result']
            source_documents=response['source_documents']
            # result_to_show=result+"\nSource Documents:\n"+str(source_documents)
            result_to_show=result
            # response="Hi , I am MedoBot"
            st.chat_message('assistant').markdown(result_to_show)
            st.session_state.messages.append({'role':'assistant','content':result_to_show})

        except Exception as e:
            st.error(f"Error : {str(e)}")
if __name__ == "__main__":
    main()