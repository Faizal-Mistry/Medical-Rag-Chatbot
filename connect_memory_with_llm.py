from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import os
from langchain_community.vectorstores import FAISS



# Step 1 : Setup LLM (GROQ)
load_dotenv()
GROQ_API_KEY=os.environ.get("GROQ_API_KEY")
def load_llm():
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=GROQ_API_KEY,      # ✅ pass API key here
        temperature=0.5,
        # max_output_tokens=512       # ✅ Groq uses max_output_tokens, not max_length
    )
    return llm


# Step 2 : Connect LLM with FAISS and create chain

custom_prompt_template="""
Use the pieces of information provided in the context to answer user's question.
If you dont know the answer, just say that you dont know, dont try to make up an answer. 
Dont provide anything out of the given context

Context: {context}
Question: {question}

Start the answer directly. No small talk please."""

def set_custom_prompt(custom_prompt_template):
    prompt=PromptTemplate(template=custom_prompt_template,input_variables=["context","question"])
    return prompt

# Load Vectore Store
DB_FAISS_PATH="vectorstore/db_faiss"
embedding_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db=FAISS.load_local(DB_FAISS_PATH,embedding_model,allow_dangerous_deserialization=True)


# create QA Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=load_llm(),   # your LLM loader (e.g., ChatGoogleGenerativeAI)
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={'k': 3}),
    return_source_documents=True,   # typo fixed
    chain_type_kwargs={
        "prompt": set_custom_prompt(custom_prompt_template)  # custom prompt
    }
)

# Now invoke with a single query

user_query=input("Write your qurery here :")
response=qa_chain.invoke({'query':user_query})
print("RESULT : ",response['result'])
print("SOURCE DOCUMENTS : ",response['source_documents'])