import os

from langchain_community.document_loaders import UnstructuredPDFLoader, DirectoryLoader, TextLoader, WebBaseLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate


GROQ_API_KEY = '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'

os.environ["GROQ_API_KEY"] = GROQ_API_KEY

loader = WebBaseLoader(['https://stoxbox.in/trading-stoxbox',
                        'https://stoxbox.in/stock-recommendations-stoxcalls',
                        'https://stoxbox.in/trading-faqs',
                        'https://stoxbox.in/stock-recommendations-stoxcalls',
                        'https://stoxbox.in/zero-brokerage'])
# loader = DirectoryLoader("stoxbox_faq/", glob="./*.txt", loader_cls=TextLoader)
documents = loader.load()

text_splitter = CharacterTextSplitter(
    chunk_size=5000,
    chunk_overlap=500
)

text_chunks = text_splitter.split_documents(documents)

# persist_directory = "faq_db"
persist_directory = "faq_db_web"

embedding = HuggingFaceEmbeddings()

vectordb = Chroma.from_documents(
    documents=text_chunks,
    embedding=embedding,
    persist_directory=persist_directory
)

# load the chroma db
# vectordb = Chroma(
#     persist_directory=persist_directory,
#     embedding_function=embedding
# )

retriever = vectordb.as_retriever()

llm = ChatGroq(
    model="llama3-70b-8192",
    temperature=0.5
)

# Load the base prompt
with open('C:/Users/KANESH/OneDrive/Desktop/chatbot/chatbot/chatbot_prompt.txt', 'r') as file:
    base_prompt = file.read()

prompt = PromptTemplate(
    template=base_prompt,
    input_variables=[
        'context',
        'question',
    ]
)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs= { 
        "prompt": prompt
    }
)

while True:
    # Get the user's question
    question = input("(Stoxbox AI) Ask a question: ")

    if question:
        response = qa_chain.invoke({"query": question})
        print(response["result"])                           