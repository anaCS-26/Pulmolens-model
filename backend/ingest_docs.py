import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

# 1. Load your secret API keys from the .env file
load_dotenv()

def populate_database_with_pdfs():
    pdf_folder = "data/pdfs"
    
    print(f"1. Scanning {pdf_folder} for PDF documents...")
    # Load all PDFs in the folder (even 100+ page ones!)
    loader = PyPDFDirectoryLoader(pdf_folder)
    raw_documents = loader.load()

    if not raw_documents:
        print(f"⚠️ No PDFs found in {pdf_folder}/. Please place your PDFs there and try again.")
        return

    print(f"   Loaded {len(raw_documents)} pages from your PDFs.")

    print(f"2. Splitting pages into manageable chunks...")
    # For large PDFs, we must split the text. If we don't, we will hit token limits for the LLM context window.
    # 1000 characters is roughly 1-2 paragraphs
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,    # Good chunk size for semantic search
        chunk_overlap=150   # Overlap ensures context isn't lost at the end of a sentence
    )
    documents = text_splitter.split_documents(raw_documents)
    print(f"   Created {len(documents)} searchable chunks.")

    print("3. Generating Gemini Embeddings & Uploading to Pinecone...")
    # Generate vectors using Gemini (Dimensions: 768)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2-preview")
    
    # Must match the Index Name you created on Pinecone
    index_name = "pulmolens-guidelines"

    # Push to Pinecone Cloud
    PineconeVectorStore.from_documents(
        documents, 
        embeddings, 
        index_name=index_name
    )
    print("✅ Success! Your 100-page PDFs have been processed and uploaded to the Pinecone database.")

if __name__ == "__main__":
    populate_database_with_pdfs()
