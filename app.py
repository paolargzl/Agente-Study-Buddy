import os
import streamlit as st
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities.wikipedia import WikipediaAPIWrapper
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun

# LangChain imports (compatibles con varias versiones)
try:
    from langchain.agents import AgentExecutor
except ImportError:
    # fallback (algunas versiones lo mueven)
    from langchain.agents.agent import AgentExecutor

try:
    from langchain.agents import create_react_agent
except ImportError:
    # fallback (rutas alternativas según versión)
    from langchain.agents.react.agent import create_react_agent

from langchain_core.prompts import PromptTemplate

load_dotenv()

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
    st.session_state["GOOGLE_API_KEY"] = st.text_input(
        "GOOGLE_API_KEY", value=default_key, type="password"
    )

    st.divider()
    st.subheader("🧠 Memoria")
    use_memory = st.toggle("Guardar historial en la sesión", value=True)

    st.divider()
    if st.button("🧹 Reset conversación", use_container_width=True):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def build_llm():
    key = st.session_state.get("GOOGLE_API_KEY") or ""
    if not key:
        st.info("👈 Pega tu GOOGLE_API_KEY en el sidebar para empezar.")
        st.stop()

    return ChatGoogleGenerativeAI(
        model=st.session_state.get("model", "gemini-2.5-flash"),
        temperature=st.session_state.get("temperature", 0.2),
        api_key=key,
    )

def build_tools():
    tools = []
    try:
        tools.append(DuckDuckGoSearchRun())
    except ImportError:
        st.warning("DuckDuckGo no disponible. Añade 'ddgs' a requirements.txt")

    wiki = WikipediaQueryRun(
        api_wrapper=WikipediaAPIWrapper(top_k_results=3, doc_content_chars_max=1500)
    )
    tools.append(wiki)
    return tools

# ✅ ReAct prompt (muy compatible y estable en Cloud)
REACT_TEMPLATE = """Eres StudyBuddy, un asistente de estudio.
Responde en español, claro y estructurado.

Tienes estas herramientas:
{tools}

Usa herramientas SOLO si las necesitas.
Si puedes responder sin buscar, responde directamente.

Formato:
Question: pregunta
Thought: tu razonamiento breve
Action: herramienta (si la usas)
Action Input: entrada
Observation: resultado
Final: respuesta final clara

Question: {input}
{agent_scratchpad}
"""
react_prompt = PromptTemplate.from_template(REACT_TEMPLATE)

def build_executor():
    llm = build_llm()
    tools = build_tools()
    agent = create_react_agent(llm, tools, react_prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=False)

# UI: historial
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

user_input = st.chat_input("Pregunta algo… (ej: '¿Qué fue la Revolución Francesa?')")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    executor = build_executor()
    chat_history = st.session_state.chat_history if use_memory else []

    with st.chat_message("assistant"):
        with st.spinner("Buscando / razonando..."):
            try:
                # metemos memoria como texto para ReAct (simple y estable)
                history_txt = ""
                for msg in chat_history[-10:]:
                    if isinstance(msg, HumanMessage):
                        history_txt += f"Usuario: {msg.content}\n"
                    else:
                        history_txt += f"Asistente: {msg.content}\n"
                full_input = user_input if not history_txt else f"{history_txt}\nPregunta: {user_input}"

                result = executor.invoke({"input": full_input})
                answer = result["output"]
            except Exception as e:
                answer = f"⚠️ Error: {e}"

        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
    if use_memory:
        st.session_state.chat_history.append(HumanMessage(content=user_input))
        st.session_state.chat_history.append(AIMessage(content=answer))
