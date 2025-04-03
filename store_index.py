from src.helper import load_pdf,text_split,huggingfaceembedding
from langchain_community.vectorstores import FAISS
extracted_data=load_pdf(r"artifacts\\Medical_book.pdf")
text_chunks=text_split(extracted_data)
embeddings=huggingfaceembedding()
def Database(text_chunks=text_chunks,embeddings=embeddings):
    database = FAISS.from_documents(text_chunks, embeddings)
    return database
