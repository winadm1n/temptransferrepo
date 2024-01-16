import os
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import CTransformers
from langchain.prompts import PromptTemplate
from flask import Flask, request, jsonify

# Load environment variables from .env file
load_dotenv('.env')

# Initializes your app with your bot token and socket mode handler
app = App(token=os.environ.get("SLACK_BOT_TOKEN"))

print("Loading model Started")

template = """only Use the following pieces of context to answer the question at the end. 
dont summarize the answer.
If you don't know the answer, just say that you don't know, don't try to make up an answer. 
{context}
Question: {question}
"""


def create_retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                           chain_type="stuff",
                                           retriever=retriever,
                                           return_source_documents=True,
                                           chain_type_kwargs={
                                               "prompt": PromptTemplate(
                                                   template=template,
                                                   input_variables=[
                                                       "context", "question"],
                                               ),
                                           },)
    print("QA chain created")
    return qa_chain


embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"device": "cpu"})
vectordb = Chroma(persist_directory=r'C:\Users\lokesh.kadi\Desktop\OnPrem Intilligent search\llama2-chat-with-documents\db',
                  embedding_function=embedding)
retriever = vectordb.as_retriever(search_kwargs={"k": 3})
print("Vectordb Loaded")
llm = CTransformers(
    model=r"C:\Users\lokesh.kadi\Desktop\OnPrem Intilligent search\llama2-chat-with-documents\model\llama-2-7b-chat.ggmlv3.q8_0.bin",
    model_type="llama",
    config={'max_new_tokens': 1200, 'temperature': 0, 'context_length': 2000},
    # max_new_tokens=600,
    temperature=0,  # type: ignore
)


def getanswer(query):
    docs = retriever.get_relevant_documents(query)
    res = qa_chain({"input_documents": docs, "query": query})
    print(res['result'])
    return res['result']


print("initiation finished")


def res():
    import time
    time.sleep(10)
    return "llama time is problem"


@app.message(".*")
def message_handler(message, say, logger):
    print(message['text'])
    # say("wait for a sec Analysing your question......")
    output = getanswer(message['text'])
    try:
        output = output.split(":")[1]
    except:
        pass
    say(str(output))
    # output = chatgpt_chain.predict(human_input = message['text'])
    # if message['text'] == "hi":
    # say("problem is with llm time")
    # say(output)


@app.event("app_mention")
def handle_app_mention_events(body, say, logger):
    # print(body)
    print(body['event']['blocks'][0]['elements'][0]['elements'][1]['text'])
    if "hi" in body['event']['blocks'][0]['elements'][0]['elements'][1]['text']:
        say("hello lokesh Test Success u can integrate backend with code")
    logger.info(body)


# Start your app
if __name__ == "__main__":
    SocketModeHandler(app, os.environ["SLACK_APP_TOKEN"]).start()
