from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnableLambda, RunnablePassthrough
import re
from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv


load_dotenv()


model = ChatGoogleGenerativeAI(
    model= "gemini-2.0-flash",
    temperature=0.2,
)

loader = PyPDFLoader("/home/ashu/langchain/final_chat.pdf")
docs = loader.load()

full_text = "\n".join([doc.page_content for doc in docs])

# Step 3: Split Q&A pairs based on the "Q1:", "Q2:", etc.
pattern = r"(Q\d+:.*?)(?=Q\d+:|$)"  # Match Q&A blocks non-greedily
matches = re.findall(pattern, full_text, flags=re.DOTALL)

# Step 4: Create a Document object for each Q&A
qa_docs = [Document(page_content=match.strip()) for match in matches]

# Step 5 (Optional): Further split long Q&A using CharacterTextSplitter
splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
result = splitter.split_documents(qa_docs)

# for i in final_docs:
#     print(i.page_content)
#     print("------------")

# loader = PyPDFLoader("/home/ashu/langchain/final_chat.pdf")
# docs = loader.load()

# for i in docs:
#     print(docs[0].page_content)
#     print("------------")

# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=500,
#     chunk_overlap=0,
# )

# text_splitter = CharacterTextSplitter(
#     separator="\n\n",
#     chunk_size=500,
#     chunk_overlap=0
# )

# result = text_splitter.split_documents(docs)

# for i in result:
#     print(result[0].page_content)
#     print("------------")

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vector_store = FAISS.from_documents(result, embeddings)

retriever = vector_store.as_retriever(search_kwargs={"k": 1})

prompt = PromptTemplate(
    template="""
      You are a helpful chatbot assistant.
      Answer ONLY from the provided transcript context.
      If the context is insufficient, just say you don't know.

      {context}
      Question: {question}
    """,
    input_variables = ['context', 'question']
)

retrieved_docs    = retriever.invoke('question')

def format_docs(retrieved_docs):
  context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
  return context_text

parallel_chain = RunnableParallel({
    'context': retriever | RunnableLambda(format_docs),
    'question': RunnablePassthrough()
})

parser = StrOutputParser()

main_chain = parallel_chain | prompt | model | parser

result1 = main_chain.invoke('my data is disapperared after update the new version of the app')
print(result1)








