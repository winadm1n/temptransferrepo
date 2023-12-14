import os

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredFileLoader
from langchain.vectorstores import Chroma

ABS_PATH: str = os.path.dirname(os.path.abspath(__file__))
DB_DIR: str = os.path.join(ABS_PATH, "db")


# Create vector database
def create_vector_database():
    """
    Creates a vector database using document loaders and embeddings.

    This function loads data from PDF, markdown and text files in the 'data/' directory,
    splits the loaded documents into chunks, transforms them into embeddings using HuggingFace,
    and finally persists the embeddings into a Chroma vector database.

    """

    # loader = DirectoryLoader(r"""C:\Users\lokesh.kadi\Desktop\OnPrem Intilligent search\llama2-chat-with-documents\data""", loader_cls=UnstructuredFileLoader, show_progress=True, use_multithreading=True)

    Confluence_URL = "https://kekarekomal.atlassian.net/wiki/"
    Username = "komal.kekare@brillio.com"
    Space_Key = "SPKey12345"
    Api_Key = "ATATT3xFfGF0i_aJjXtIiHWvsdVkwuahSl9nCgF0N4OQvhItxB7Mb_se_gL6XBnyzYC2RkpiO5DEiHIaISQHALKuI5cgtcbYnZS6X3Ms1gWvsqCYaeAXsm2-nfQzT24qEHyAgq8vQPNAc6lnEQTXNH7rJ4INCWZblzbS7Da8GvcO9DPAAj5NaoU=DCB9C82E"
    from langchain.document_loaders import ConfluenceLoader

    loader = ConfluenceLoader(
        url=Confluence_URL, username=Username, api_key=Api_Key)
    loaded_documents = loader.load(
        space_key=Space_Key, include_attachments=True, limit=50)

    # loaded_documents = loader.load()

    # Split loaded documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300, chunk_overlap=30)
    chunked_documents = text_splitter.split_documents(loaded_documents)

    # Initialize HuggingFace embeddings
    huggingface_embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )

    # Create and persist a Chroma vector database from the chunked documents
    vector_database = Chroma.from_documents(
        documents=chunked_documents,
        embedding=huggingface_embeddings,
        persist_directory=DB_DIR,
    )

    vector_database.persist()


if __name__ == "__main__":
    create_vector_database()
