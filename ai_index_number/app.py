from __future__ import annotations

import time

import streamlit as st
from dotenv import load_dotenv

from src.generation.llm_client import OllamaClient
from src.pipeline.rag_pipeline import AcademicCityRAG

load_dotenv()

# Student metadata placeholder: Name=<YOUR_NAME_HERE>, Index=<YOUR_INDEX_NUMBER_HERE>
st.set_page_config(page_title="Academic City Assistant", layout="wide")


def init_state() -> None:
    if "rag" not in st.session_state:
        st.session_state.rag = AcademicCityRAG()
    if "llm" not in st.session_state:
        st.session_state.llm = OllamaClient()
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "runs" not in st.session_state:
        st.session_state.runs = []
    if "started" not in st.session_state:
        st.session_state.started = False
    if "landing_query" not in st.session_state:
        st.session_state.landing_query = ""


def apply_styles() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(1100px 520px at 18% 0%, rgba(56, 189, 248, 0.14), transparent 60%),
                radial-gradient(900px 480px at 88% 10%, rgba(16, 185, 129, 0.14), transparent 62%),
                linear-gradient(180deg, #0b1220 0%, #0a1f1c 52%, #0f172a 100%);
        }
        .block-container {padding-top: 1.1rem; max-width: 1120px;}
        .landing-wrap, .hero-card, .panel-card {
            border-radius: 20px;
            border: 1px solid rgba(148, 163, 184, 0.24);
            box-shadow: 0 12px 30px rgba(0,0,0,0.24);
            backdrop-filter: blur(6px);
        }
        .landing-wrap {
            background: linear-gradient(140deg, rgba(30, 41, 59, 0.78), rgba(15, 23, 42, 0.86));
            padding: 42px 34px 34px 34px;
            margin-top: 8vh;
            animation: fadeUp 500ms ease-out;
            text-align: center;
        }
        .hero-card {
            background: linear-gradient(135deg, rgba(30,41,59,0.72), rgba(15,23,42,0.85));
            padding: 18px 20px;
            margin-bottom: 16px;
            animation: fadeUp 350ms ease-out;
        }
        .panel-card {
            background: rgba(15, 23, 42, 0.50);
            padding: 12px;
            margin-bottom: 10px;
        }
        .landing-title, .hero-title {
            color: #f8fafc;
            font-weight: 750;
            letter-spacing: 0.1px;
        }
        .landing-title {
            font-size: clamp(2rem, 4.2vw, 3.2rem);
            line-height: 1.08;
            margin-bottom: 0.55rem;
        }
        .hero-title {font-size: 1.5rem; margin-bottom: 0.22rem;}
        .landing-sub, .hero-sub {
            color: #cbd5e1;
            font-size: 1.05rem;
            margin-bottom: 0.3rem;
        }
        .landing-desc {
            color: #94a3b8;
            font-size: 1rem;
            margin-top: 0.95rem;
            margin-bottom: 1.2rem;
            line-height: 1.45;
            max-width: 760px;
            margin-left: auto;
            margin-right: auto;
        }
        .cta-help {color: #a7f3d0; font-size: 0.88rem; margin-top: 0.75rem;}
        .score-pill {
            background: linear-gradient(135deg, rgba(16,185,129,0.27), rgba(34,197,94,0.18));
            color: #d1fae5;
            border: 1px solid rgba(110,231,183,0.33);
            border-radius: 999px;
            padding: 0.22rem 0.72rem;
            display: inline-block;
            margin-right: 0.3rem;
            margin-bottom: 0.32rem;
            font-size: 0.82rem;
        }
        .mini-card {
            background: rgba(2, 44, 34, 0.62);
            border: 1px solid rgba(110, 231, 183, 0.25);
            border-radius: 14px;
            padding: 10px 12px;
            margin-bottom: 8px;
        }
        .mini-title {color: #a7f3d0; font-size: 0.8rem;}
        .mini-value {color: #ecfeff; font-size: 1.12rem; font-weight: 700;}
        div[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0b1220 0%, #0b1f1a 100%);
            border-right: 1px solid rgba(148,163,184,0.15);
        }
        div[data-testid="stChatMessage"] {
            background: rgba(15, 23, 42, 0.58);
            border: 1px solid rgba(148, 163, 184, 0.22);
            border-radius: 14px;
            padding: 6px;
        }
        div[data-testid="stTextInput"] input {
            border-radius: 14px !important;
            background: rgba(15, 23, 42, 0.86) !important;
            border: 1px solid rgba(148, 163, 184, 0.42) !important;
            color: #f8fafc !important;
            padding: 0.72rem 0.95rem !important;
        }
        .stButton > button {
            border-radius: 12px !important;
            border: 1px solid rgba(148,163,184,0.45) !important;
            box-shadow: 0 8px 18px rgba(0,0,0,0.22) !important;
            transition: transform 0.15s ease, box-shadow 0.15s ease !important;
        }
        .stButton > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 10px 22px rgba(0,0,0,0.28) !important;
        }
        @keyframes fadeUp {
            from {opacity: 0; transform: translateY(10px);}
            to {opacity: 1; transform: translateY(0);}
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# --------------------------- Sidebar Controls ---------------------------
def render_sidebar() -> tuple[int, str, str, bool]:
    with st.sidebar:
        st.header("⚙️ Chat Controls")
        top_k = st.slider("top_k retrieval", min_value=4, max_value=15, value=8)
        prompt_version = st.selectbox("Prompt version", ["v1", "v2", "v3"], index=2)
        chunk_method = st.selectbox("Chunking method", ["paragraph", "fixed"], index=0)
        debug_mode = st.toggle("Debug mode", value=True)
        st.caption("Use v3 for strongest grounded responses.")

        if st.button("🔁 Rebuild Index", use_container_width=True):
            st.session_state.rag = AcademicCityRAG(chunk_method=chunk_method)
            st.success("Index rebuilt.")

        if st.button("🧹 Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.runs = []
            st.success("Chat cleared.")
    return top_k, prompt_version, chunk_method, debug_mode


def handle_query(user_text: str, top_k: int, prompt_version: str) -> None:
    st.session_state.messages.append({"role": "user", "content": user_text})
    with st.chat_message("user", avatar="🧑‍🎓"):
        st.markdown(user_text)

    with st.chat_message("assistant", avatar="🤖"):
        placeholder = st.empty()
        for dots in [".", "..", "..."]:
            placeholder.markdown(f"Thinking with retrieval + scoring{dots}")
            time.sleep(0.12)
        with st.spinner("Generating response..."):
            result = st.session_state.rag.answer(
                query=user_text,
                llm_client=st.session_state.llm,
                top_k=top_k,
                prompt_version=prompt_version,
            )
        placeholder.markdown(result["answer"])

    st.session_state.messages.append({"role": "assistant", "content": result["answer"]})
    st.session_state.runs.append(result)


# --------------------------- Landing Page ---------------------------
def render_landing_page(top_k: int, prompt_version: str) -> None:
    st.markdown(
        """
        <div class="landing-wrap">
            <div class="landing-title">The future of academic intelligence starts here</div>
            <div class="landing-sub">Ask questions about Ghana elections and the 2025 budget using AI-powered retrieval.</div>
            <div class="landing-desc">
                Built for fast grounded answers, transparent scoring, and evidence-driven reasoning.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, center, right = st.columns([1.2, 5.6, 1.2])
    with center:
        input_col, ask_col = st.columns([4.7, 1.3])
        with input_col:
            hero_query = st.text_input(
                "Ask AI",
                key="landing_query_input",
                label_visibility="collapsed",
                placeholder="Ask a question...",
            )
        with ask_col:
            ask_clicked = st.button("Ask AI", use_container_width=True, type="primary")

        cta_left, cta_mid, cta_right = st.columns([1.9, 2.2, 1.9])
        with cta_mid:
            explore_clicked = st.button("Explore Dataset", use_container_width=True)

        st.markdown("<div class='cta-help'>Press Enter in the input or click Ask AI to start.</div>", unsafe_allow_html=True)

        if ask_clicked and hero_query.strip():
            st.session_state.landing_query = hero_query.strip()
            st.session_state.started = True
            st.rerun()

        if explore_clicked:
            st.session_state.landing_query = "What key datasets are available in this system?"
            st.session_state.started = True
            st.rerun()


# --------------------------- Chat Interface ---------------------------
def render_chat(top_k: int, prompt_version: str) -> None:
    st.markdown(
        """
        <div class="hero-card">
          <div class="hero-title">Academic City Assistant</div>
          <div class="hero-sub">Ghana Data Intelligence</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    for msg in st.session_state.messages:
        avatar = "🧑‍🎓" if msg["role"] == "user" else "🤖"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])

    query = st.chat_input("Ask about elections or budget data...")
    if query and query.strip():
        handle_query(query.strip(), top_k=top_k, prompt_version=prompt_version)


# --------------------------- Debug Panels ---------------------------
def render_debug_panels(prompt_version: str, debug_mode: bool) -> None:
    if not st.session_state.runs:
        return

    latest = st.session_state.runs[-1]
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            f"<div class='mini-card'><div class='mini-title'>Query Type</div><div class='mini-value'>{latest['query_type'].title()}</div></div>",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"<div class='mini-card'><div class='mini-title'>Retrieved Chunks</div><div class='mini-value'>{len(latest['retrieved_chunks'])}</div></div>",
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"<div class='mini-card'><div class='mini-title'>Selected Context</div><div class='mini-value'>{len(latest['selected_context'])}</div></div>",
            unsafe_allow_html=True,
        )

    st.subheader("📚 Retrieved Chunks")
    for ch in latest["retrieved_chunks"]:
        with st.expander(f"{ch['chunk_id']} • {ch['source']} • final={ch['final_score']:.3f}"):
            st.markdown(
                (
                    f"<span class='score-pill'>Vector: {ch['vector_score']:.3f}</span>"
                    f"<span class='score-pill'>BM25: {ch['bm25_score']:.3f}</span>"
                    f"<span class='score-pill'>Final: {ch['final_score']:.3f}</span>"
                ),
                unsafe_allow_html=True,
            )
            st.write(ch["text"][:1700] + ("..." if len(ch["text"]) > 1700 else ""))

    st.subheader("🧩 Selected Context")
    for ch in latest["selected_context"]:
        st.markdown(f"- `{ch['chunk_id']}` ({ch['source']}) score={ch['final_score']:.3f}")

    if debug_mode:
        with st.expander("🛠️ Debug: Final Prompt", expanded=False):
            st.code(latest["final_prompt"], language="text")

        with st.expander("ℹ️ Debug: Pipeline State", expanded=False):
            st.write(f"Query type: {latest['query_type']}")
            st.write(f"Prompt version: {prompt_version}")


def main() -> None:
    init_state()
    apply_styles()
    top_k, prompt_version, chunk_method, debug_mode = render_sidebar()

    if not st.session_state.started:
        render_landing_page(top_k=top_k, prompt_version=prompt_version)
        return

    if st.session_state.landing_query.strip():
        pending = st.session_state.landing_query.strip()
        st.session_state.landing_query = ""
        handle_query(pending, top_k=top_k, prompt_version=prompt_version)

    render_chat(top_k=top_k, prompt_version=prompt_version)
    render_debug_panels(prompt_version=prompt_version, debug_mode=debug_mode)


if __name__ == "__main__":
    main()
