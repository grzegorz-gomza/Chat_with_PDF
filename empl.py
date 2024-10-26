from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.memory import ConversationBufferMemory
from langchain_openai import OpenAI
from langchain.chains import ConversationalRetrievalChain
import openai
import autogen
import os




#set llm for langchain using model from lmstudio
openai.api_type = "open_ai"
openai.base_url = "http://localhost:1234/v1"
openai.api_key = "NULL"

os.environ["OPENAI_API_KEY"] = "NULL"
os.environ["BASE_URL"]="http://localhost:1234/v1"
os.environ["API_TYPE"] = "open_ai"

client = OpenAI(base_url="http://localhost:1234/v1", api_key="not-needed")

#load the pdf file from directory
loaders = [PyPDFLoader('./chat_docs.pdf')]
docs = []
for file in loaders:
    docs.extend(file.load())
#split text to chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
docs = text_splitter.split_documents(docs)

#create a vectorstore
vectorstore = Chroma(
collection_name="full_documents",
embedding_function=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", 
                                           model_kwargs={'device': 'cpu'})
)
vectorstore.add_documents(docs)

qa = ConversationalRetrievalChain.from_llm(
    llm = client,
    retriever = vectorstore.as_retriever(),
    memory=ConversationBufferMemory(memory_key="chat_history", return_messages=True)
)


#set config for autogen
config_list = [
    {
         "api_base": "http://localhost:1234/v1",
        "api_key": "NULL"
    }
]

#set autogen user agent and assistant agent with function calling
llm_config={
    "request_timeout": 600,
    "seed": 42,
    "config_list": config_list,
    "temperature": 0,
    "functions": [
        {
            "name": "chat_docs",
            "description": "Answer any chat_docs related questions",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The question to ask in relation to chat_docs",
                    }
                },
                "required": ["question"],
            },
        }
    ],
}

#the function takes a parameter question,calls the qa chain and answer it by returnin the answer
# from the chain
def chat_docs(question):
    response = qa({"question": question})
    return response["answer"]


# create an AssistantAgent instance "assistant"
assistant = autogen.AssistantAgent(
    name="assistant",
    llm_config=llm_config,
)


# create a UserProxyAgent instance "user_proxy"
user_proxy = autogen.UserProxyAgent(
    name="user_proxy",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=1,
    code_execution_config={"work_dir": "docs"},
    llm_config=llm_config,
    system_message="""Reply TERMINATE if the task has been solved at full satisfaction.
Otherwise, reply CONTINUE, or the reason why the task is not solved yet.""",
    function_map={"chat_docs":chat_docs}
)

# the assistant receives a message from the user, which contains the task description
user_proxy.initiate_chat(
    assistant,
    message="""
Find the answers to the 3 questions below from the chat_docs.pdf and do not write any code.

1.Who is the CEO of walmart?
2.What did Doug McMillon write about walmart.
3.Write about the Executive Shuffle?

Start the work now.
"""
)
