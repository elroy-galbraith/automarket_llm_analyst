import streamlit as st
import duckdb
from langchain_ollama.chat_models import ChatOllama
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain.tools import tool
from langchain_core.messages import AIMessage, HumanMessage
from tenacity import retry, stop_after_attempt, wait_fixed

llm = ChatOllama(model="llama3.1")
#Create temp table
duckdb.query("""CREATE OR REPLACE TEMPORARY TABLE jamaican_car_listings
              as
            Select * from 'data.parquet' where price > 50000;""")

table_name = 'jamaican_car_listings'
schema = str(duckdb.query("Select * from jamaican_car_listings limit 5;").description)

st.title("AutoAnalyzerJA 🚗📈🤖")

# Initialize chat history
if "messages" not in st.session_state:
    with st.chat_message("assistant"):
        st.markdown("Hello! How can I help you in analyzing the Jamaican car market?")
    st.session_state.messages = []
    st.session_state.thread_id = None


def update_sidebar():
    st.sidebar.header("Query")
    st.sidebar.code(st.session_state.query, language="SQL")
    st.sidebar.header("Output")
    st.sidebar.code(st.session_state.query_output)

if "query" not in st.session_state:
    st.session_state.query = ""
    st.session_state.query_output = ""

st.session_state.update_sidebar = update_sidebar

@tool
def execute_query(query: str) -> str:
    """Takes SQL query (for duckdb database) and executes it on the server"""
    try:
        output = str(duckdb.query(query))
        st.session_state.query = query
        st.session_state.query_output = output
        return output
    except Exception as e:
        return str(e)

@retry(stop=stop_after_attempt(5), wait=wait_fixed(1), reraise=True)
def invoke_agent_with_retry(prompt):
    return agent_executor.invoke({"input": prompt,
                                  "chat_history": convertToLangChainMessages(st.session_state.messages)},
                                  handle_parsing_errors=True, stream=False)

tools = [execute_query]
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            f"""<overview>You are a highly paid data analyst, who is particularly proficient in SQL.
                         Your job is to take user queries and transform them into a usable SQL query that will answer their question</overview>
                        <relevant rule>You must __always__ use the execute_query tool</relevant rules>
                        <relevant rule>Your SQL must be precisely correct and follow all syntax rules typical of DuckDB</relevant rule>
                        <relevant rule>You can sometimes expect not to get an exact match in your queries so controlling for case is helpful</relevant rule>
                        <relevant rule>Often you will want to do fuzzy searches like using the % symbol to control for slight differences</relevant rule>
                        <relevant rule>You should __interpret__ the results of the queries and respond naturally to the user</releant rule>
                        <relevant rule>You can simply omit the explanation of the query in your **final** response</relevant rule>
                        <IMPERATIVE><relevant rule>Your final answer should make NO REFERENCE to the query</relevant answer></IMPERATIVE>
                        <table_name>{table_name}</table_name>
                        <schema>{schema}</schema>"""
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)

# Construct the Tools agent
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True,)

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    role = message["role"]
    with st.chat_message(role):
        st.markdown(message['message'])

def convertToLangChainMessages(history):
    messages = []
    for message in history:
        if message['role'] == "User":
            messages.append(HumanMessage(content=message["message"]))
        else:
             messages.append(AIMessage(content=message["message"]))
    return messages

if prompt := st.chat_input("What's the most popular car make and model?"):
    st.session_state.messages.append({
        "role": "User",
        "message": prompt
    })
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        try:
            response = invoke_agent_with_retry(prompt)
            output = response['output']
            message_placeholder.markdown(output)
        except Exception as e:
            print(e)
            output = "Failed to process your request. Please try again."
            message_placeholder.error(output)
        st.session_state.update_sidebar()
    st.session_state.messages.append({
        "role": "Assistant",
        "message": output
    })