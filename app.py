import os
import streamlit as st
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities.wikipedia import WikipediaAPIWrapper
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun

load_dotenv()

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="StudyBuddy — Gemini Search Agent", page_icon="🔎", layout="wide")
st.title("🔎 StudyBuddy — Gemini + LangChain Agent")
st.caption("Agente con tools: DuckDuckGo Search + Wikipedia. Pega tu API key en el sidebar.")

with st.sidebar:
    st.header("⚙️ Ajustes")

    st.session_state["model"] = st.selectbox(
        "Modelo Gemini",
        ["gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-2.5-pro"],
        index=0,
        key="model_select"
    )

    st.session_state["temperature"] = st.slider(
        "Temperatura", 0.0, 1.0, 0.2, 0.05, key="temp_slider"
    )


    st.divider()
    st.subheader("🔑 API Key (Gemini)")
    default_key = os.getenv("GOOGLE_API_KEY", "")
    ui_key = st.text_input(
        "GOOGLE_API_KEY",
        value=default_key,
        type="password",
        help="Key de Google AI Studio (formato AIza...). No se sube a GitHub."
    )
    st.session_state["GOOGLE_API_KEY"] = ui_key

    st.divider()
    st.subheader("🧠 Memoria")
    use_memory = st.toggle("Guardar historial en la sesión", value=True)

    st.divider()
    if st.button("🧹 Reset conversación", use_container_width=True):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.rerun()

# Estado
if "messages" not in st.session_state:
    st.session_state.messages = []      # para UI
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # para LangChain

def build_llm():
    key = st.session_state.get("GOOGLE_API_KEY") or ""
    if not key:
        st.info("👈 Pega tu GOOGLE_API_KEY en el sidebar para empezar.")
        st.stop()

    model_name = st.session_state.get("model") or "gemini-2.5-flash"
    temp = st.session_state.get("temperature", 0.2)

    return ChatGoogleGenerativeAI(
        model=model_name,
        temperature=temp,
        api_key=key,
    )



def build_tools():
    ddg = DuckDuckGoSearchRun()
    wiki = WikipediaQueryRun(
        api_wrapper=WikipediaAPIWrapper(top_k_results=3, doc_content_chars_max=1500)
    )
    return [ddg, wiki]

def build_tools():
    tools = []

    try:
        ddg = DuckDuckGoSearchRun()
        tools.append(ddg)
    except ImportError:
        st.warning("DuckDuckGo no disponible (falta ddgs). Instala: python3 -m pip install -U ddgs")

    wiki = WikipediaQueryRun(
        api_wrapper=WikipediaAPIWrapper(top_k_results=3, doc_content_chars_max=1500)
    )
    tools.append(wiki)

    return tools

SYSTEM = """Eres StudyBuddy, un asistente de estudio.

Responde en español, claro y estructurado.

Reglas IMPORTANTES:
- SOLO usa herramientas si realmente necesitas buscar información externa.
- Si ya sabes la respuesta, responde directamente SIN usar tools.
- Usa DuckDuckGoSearchRun solo para información actual o noticias.
- Usa WikipediaQueryRun solo si necesitas una definición específica.

Evita decir que volverás a buscar.
Da siempre una respuesta final útil.
"""


prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

def build_agent_executor():
    llm = build_llm()
    tools = build_tools()
    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=False)

# Mostrar historial
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

user_input = st.chat_input("Pregunta algo… (ej: '¿Qué fue la Revolución Francesa?')")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    executor = build_agent_executor()
    chat_history = st.session_state.chat_history if use_memory else []

    with st.chat_message("assistant"):
        with st.spinner("Buscando / razonando..."):
            try:
                result = executor.invoke({"input": user_input, "chat_history": chat_history})
                answer = result["output"]
            except Exception as e:
                answer = f"⚠️ Error: {e}"
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})

    if use_memory:
        st.session_state.chat_history.append(HumanMessage(content=user_input))
        st.session_state.chat_history.append(AIMessage(content=answer))
