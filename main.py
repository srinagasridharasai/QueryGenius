import os
import streamlit as st
import pandas as pd
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import HuggingFaceHub
from langchain.chains import RetrievalQA
from dotenv import load_dotenv

load_dotenv()

TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
os.environ["HUGGINGFACEHUB_API_TOKEN"] = TOKEN

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
llm = HuggingFaceHub(repo_id="meta-llama/Meta-Llama-3-8B-Instruct", model_kwargs={'temperature': 0.5})

CHROMA_DB_DIR = "./chroma_db"

# Streamlit app title
st.title("CSV to Embeddings with Query Handling")

# Step 1: File upload
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        uploaded_file.seek(0)  # Rewind the file to the start
        file_content = uploaded_file.read(1024).decode("utf-8", errors="ignore")
        st.write(f"First 1KB of file: {file_content[:500]}...")
        encodings = ['utf-8', 'latin1', 'ISO-8859-1', 'cp1252']
        data = None
        for encoding in encodings:
            try:
                st.write(f"Trying to read the file with {encoding} encoding...")
                uploaded_file.seek(0)  # Reset file pointer to the start
                data = pd.read_csv(uploaded_file, encoding=encoding)
                st.success(f"File loaded successfully with {encoding} encoding.")
                break
            except UnicodeDecodeError:
                st.write(f"Failed to read the file with {encoding} encoding.")
            except Exception as e:
                st.write(f"An error occurred while reading the file with {encoding}: {e}")

        if data is None or data.empty:
            st.error("Failed to read the CSV file or the file is empty.")
        else:
            st.write("Preview of Uploaded File:")
            st.dataframe(data)

            if data.empty:
                st.error("Uploaded CSV is empty.")
            else:
                content_column = st.selectbox("Select the column to use for embedding", data.columns)

                if content_column:
                    content = data[content_column].dropna().tolist()
                    st.write("Number of entries to embed:", len(content))

                    if not os.path.exists(CHROMA_DB_DIR):
                        st.write("ChromaDB directory not found. Creating new ChromaDB...")
                        os.makedirs(CHROMA_DB_DIR, exist_ok=True)

                    st.write("Generating embeddings and storing in ChromaDB...")
                    chroma_db = Chroma.from_texts(content, embedding, persist_directory=CHROMA_DB_DIR)
                    chroma_db.persist()
                    st.success("Embeddings stored in ChromaDB successfully!")

                    enable_query_mode = st.checkbox("Enable Query Mode", value=True)

                    if enable_query_mode:
                        st.write("Enter your query below:")
                        user_query = st.text_input("Query")

                        if user_query:
                            retriever = chroma_db.as_retriever()
                            qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

                            st.write("Processing your query...")
                            response = qa_chain.run(user_query)
                            st.write("Response:")
                            st.write(response)
                        else:
                            st.warning("Please enter a query to process.")
                else:
                    st.error(f"Column '{content_column}' not found in the uploaded file.")
    except Exception as e:
        st.error(f"An error occurred while processing the file: {e}")
