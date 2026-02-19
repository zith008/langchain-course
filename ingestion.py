import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_pinecone import PineconeVectorStore


load_dotenv()






if __name__ == '__main__':
    print("Ingesting...")
    # print(os.getenv("PINECONE_API"))

loader = TextLoader("mediumblog1.txt")
document = loader.load()

print("splitting...")

"""keep the chunks small enough to fit context window, 
and it should be big enough to know what this chunk means when 
read alone holds semantic meaning"""

text_splitter = CharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200)
texts = text_splitter.split_documents(document)
print(f"created {len(texts)} chunks")

embeddings = OpenAIEmbeddings()
print("Embedding...")

PineconeVectorStore.from_documents(
    texts,embeddings, 
    index_name=os.environ['INDEX_NAME'])

print("Done!")



