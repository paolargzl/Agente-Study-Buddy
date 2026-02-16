import os
import streamlit as st
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage

from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool  # ✅ (NUEVO) para envolver DDG de forma segura
from langchain_community.utilities.wikipedia import WikipediaAPIWrapper
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun

from langgraph.prebuilt import create_react_agent

load_dotenv()

st.set_page_config(page_title="StudyBuddy — Gemini Search Agent", page_icon="🔎", layout="wide")
st.title("🔎 StudyBuddy — Gemini + Agent (Stable)")
st.caption("Agente con tools: DuckDuckGo + Wikipedia. Deploy-friendly (LangGraph).")

with st.sidebar:
    st.header("⚙️ Ajustes")

    model = st.selectbox(
        "Modelo Gemini",
        ["gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-2.5-pro"],
        index=0,
    )
    temperature = st.slider("Temperatura", 0.0, 1.0, 0.2, 0.05)

    st.divider()
    st.subheader("🔑 API Key (Gemini)")
    default_key = os.getenv("GOOGLE_API_KEY", "")
    api_key = st.text_input("GOOGLE_API_KEY", value=default_key, type="password")

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
    if not api_key:
        st.info("👈 Pega tu GOOGLE_API_KEY en el sidebar (o ponla en Secrets en Streamlit Cloud).")
        st.stop()

    return ChatGoogleGenerativeAI(
        model=model,
        temperature=temperature,
        api_key=api_key,
    )

# ✅ (NUEVO) Tool segura para DDG: evita el error "Expecting value..."
_ddg_raw = DuckDuckGoSearchRun()

@tool
def duckduckgo_search(query: str) -> str:
    """Busca en DuckDuckGo. Devuelve texto con resultados. Nunca crashea (devuelve mensaje de error controlado)."""
    try:
        out = _ddg_raw.run(query)
        if out is None or not str(out).strip():
            return "DuckDuckGo no devolvió resultados (respuesta vacía o bloqueada)."
        return str(out)
    except Exception as e:
        return f"DuckDuckGo falló: {e}"

def build_tools():
    tools = []

    # ✅ (CAMBIO) usamos la tool segura en lugar de DuckDuckGoSearchRun() directo
    try:
        tools.append(duckduckgo_search)
    except Exception:
        st.warning("DuckDuckGo no disponible. Asegúrate de tener 'ddgs' en requirements.txt.")

    tools.append(
        WikipediaQueryRun(
            api_wrapper=WikipediaAPIWrapper(top_k_results=3, doc_content_chars_max=1500)
        )
    )
    return tools

# ✅ (CAMBIO PEQUEÑO) prompt más robusto para que no se quede atascado si DDG falla
SYSTEM = """Eres StudyBuddy, un asistente de estudio.
Responde en español, claro y estructurado.

Reglas:
- Usa herramientas SOLO si necesitas buscar información externa.
- Wikipedia: definiciones / explicación enciclopédica.
- DuckDuckGo: info actual / datos concretos.
- Si una herramienta falla o devuelve vacío, NO te atasques: responde con lo que sepas y sugiere cómo refinar la búsqueda.
- Da siempre una respuesta final útil.
"""

# Construimos el agent una vez por ejecución
llm = build_llm()
tools = build_tools()
agent = create_react_agent(llm, tools, prompt=SYSTEM)

# UI: historial
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

user_input = st.chat_input("Pregunta algo… (ej: '¿Qué fue la Revolución Francesa?')")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

     # Memoria: LangGraph usa lista de mensajes
    msgs = []
    if use_memory:
        msgs.extend(st.session_state.chat_history)

    msgs.append(HumanMessage(content=user_input))

    with st.chat_message("assistant"):
        with st.spinner("Buscando / razonando..."):
            try:
                result = agent.invoke({"messages": msgs})
                final_msg = result["messages"][-1]
                content = getattr(final_msg, "content", "")

                # ✅ NORMALIZAR RESPUESTA DE GEMINI
                if isinstance(content, list):
                    answer = "\n".join(
                        part.get("text", str(part)) if isinstance(part, dict) else str(part)
                        for part in content
                    ).strip()
                elif isinstance(content, dict):
                    answer = str(content.get("text", content)).strip()
                else:
                    answer = str(content).strip()

            except Exception as e:
                answer = f"⚠️ Error: {e}"

        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})

    if use_memory:
        st.session_state.chat_history.append(HumanMessage(content=user_input))
        st.session_state.chat_history.append(AIMessage(content=answer))

