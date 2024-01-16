import os

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.document_loaders import ConfluenceLoader

ABS_PATH: str = os.path.dirname(os.path.abspath(__file__))
DB_DIR: str = os.path.join(ABS_PATH, "db")


# Create vector database
def create_vector_database():
    loader = ConfluenceLoader(url="https://confluence1998.atlassian.net/wiki/",
                              username="komalkekare1998@gmail.com",
                              api_key="ATATT3xFfGF0fGW_mHDog_atR3pOUYsEfKDu_jpp68ZVQP--u_lOE_EaGxAFauH7DSyNzT9lW2BcoEkBqM2SDDoH4Cge9TdMfvGGINSdr9_IiRxGOgvWQ_1Jis35qr4ZnjDLd8oCB4XGOAsorp5UXOE7IXN-deF0jy9C4JtMhhAzLdAHdoCblUw=F3D51E2E")
    loaded_documents = loader.load(space_key="SI")
    print('done')

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400, chunk_overlap=30)
    chunked_documents = text_splitter.split_documents(loaded_documents)

    # Initialize HuggingFace embeddings
    huggingface_embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        # model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={"device": "cpu"},
    )

    # Create and persist a Chroma vector database from the chunked documents
    vector_database = Chroma.from_documents(
        documents=chunked_documents,
        embedding=huggingface_embeddings,
        persist_directory=DB_DIR,
    )

    vector_database.persist()
    print("vectordb created")


if __name__ == "__main__":
    create_vector_database()
