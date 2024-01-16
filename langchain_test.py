import os

# get parent dir
path_of_this_file = os.path.dirname(os.path.realpath(__file__))
os.chdir(os.path.join(path_of_this_file, os.pardir))

# get my api key from text file
with open("openai_api_key.txt") as f:
    data = f.read()

os.environ["OPENAI_API_KEY"] = data

if True:
    # pip install --upgrade --quiet langchain-core langchain-community langchain-openai
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_openai import ChatOpenAI

    prompt = ChatPromptTemplate.from_template("tell me a short joke about {topic}")
    model = ChatOpenAI(model="gpt-4")
    output_parser = StrOutputParser()

    chain = prompt | model | output_parser

    print(chain.invoke({"topic": "ice cream"}))

if False:
    # pip install langchain docarray tiktoken
    # pip install pydantic==1.10.13
    from langchain_community.vectorstores import DocArrayInMemorySearch
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnableParallel, RunnablePassthrough
    from langchain_openai.chat_models import ChatOpenAI
    from langchain_openai.embeddings import OpenAIEmbeddings

    vectorstore = DocArrayInMemorySearch.from_texts(
        ["harrison worked at kensho", "bears like to eat honey"],
        embedding=OpenAIEmbeddings(),
    )
    retriever = vectorstore.as_retriever()

    template = """Answer the question based only on the following context:
    {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    model = ChatOpenAI()
    output_parser = StrOutputParser()

    setup_and_retrieval = RunnableParallel(
        {"context": retriever, "question": RunnablePassthrough()}
    )
    chain = setup_and_retrieval | prompt | model | output_parser

    print(chain.invoke("where did harrison work?"))