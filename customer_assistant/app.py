"""
================================================================================
CUSTOMER ASSISTANT — STREAMLIT MVP
================================================================================

A RAG-powered customer support chatbot with a ChatGPT-like interface.

ARCHITECTURE:

    +-----------+      +------------------+      +---------------------+
    |   Chat    | ---> |  RAG Retrieval   | ---> |   Claude (Anthropic) |
    |  (User)   |      | (keyword scoring)|      |   or offline fallback|
    +-----------+      +--------+---------+      +---------------------+
                                |
                                v
                       +------------------+
                       |  Knowledge Base  |
                       | (built-in + docs |
                       |  uploaded by you)|
                       +------------------+

RAG RETRIEVAL
    Each query is scored against a curated knowledge base using bag-of-words
    overlap on document titles, tags, and body text. The top-k matches are
    injected into the system prompt as grounding context before the LLM call.

USAGE
    pip install -r requirements.txt
    export ANTHROPIC_API_KEY=sk-ant-...    # optional — offline fallback used otherwise
    streamlit run app.py
================================================================================
"""

from __future__ import annotations

import os
import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import streamlit as st

try:
    import anthropic
    _ANTHROPIC_IMPORTED = True
except Exception:
    _ANTHROPIC_IMPORTED = False


# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Customer Assistant",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# THEME — ChatGPT-like dark UI
# ============================================================
_CSS = """
<style>
/* ── Global ─────────────────────────────────────────────── */
html, body, [data-testid="stAppViewContainer"] {
    background-color: #212121;
    color: #ececec;
    font-family: "Söhne", ui-sans-serif, system-ui, -apple-system, sans-serif;
}

/* ── Sidebar ─────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background-color: #171717;
    border-right: 1px solid #2a2a2a;
}
[data-testid="stSidebar"] * { color: #ececec !important; }
[data-testid="stSidebar"] .stTextInput input {
    background-color: #2f2f2f !important;
    border-color: #444 !important;
    color: #ececec !important;
}

/* ── Main content area ───────────────────────────────────── */
[data-testid="stMain"] {
    background-color: #212121;
}

/* ── Chat messages ───────────────────────────────────────── */
[data-testid="stChatMessage"] {
    background-color: transparent !important;
    border: none !important;
    padding: 0.25rem 0 !important;
    max-width: 760px;
    margin: 0 auto;
}

/* User message bubble */
[data-testid="stChatMessage"][data-testid*="user"] {
    background-color: #2f2f2f !important;
    border-radius: 1rem !important;
    padding: 0.75rem 1rem !important;
}

/* ── Chat input ──────────────────────────────────────────── */
[data-testid="stChatInput"] {
    background-color: #2f2f2f !important;
    border: 1px solid #444 !important;
    border-radius: 0.75rem !important;
    max-width: 760px;
    margin: 0 auto;
}
[data-testid="stChatInput"] textarea {
    background-color: transparent !important;
    color: #ececec !important;
}
[data-testid="stChatInput"] button { background-color: #10a37f !important; }

/* ── Source expander ─────────────────────────────────────── */
[data-testid="stExpander"] {
    background-color: #2f2f2f !important;
    border: 1px solid #3a3a3a !important;
    border-radius: 0.5rem !important;
}

/* ── Buttons ─────────────────────────────────────────────── */
div.stButton > button[kind="primary"] {
    background-color: #10a37f !important;
    border: none !important;
    color: white !important;
    border-radius: 0.5rem !important;
}
div.stButton > button:not([kind="primary"]) {
    background-color: #2f2f2f !important;
    border: 1px solid #444 !important;
    color: #ececec !important;
    border-radius: 0.5rem !important;
}

/* ── Headings / captions ─────────────────────────────────── */
h1, h2, h3, h4 { color: #ececec !important; }
.stCaption { color: #8e8ea0 !important; }

/* ── Divider ─────────────────────────────────────────────── */
hr { border-color: #2a2a2a !important; }

/* ── File uploader ───────────────────────────────────────── */
[data-testid="stFileUploader"] {
    background-color: #2f2f2f !important;
    border: 1px dashed #444 !important;
    border-radius: 0.5rem !important;
}
</style>
"""
st.markdown(_CSS, unsafe_allow_html=True)


# ============================================================
# CONSTANTS
# ============================================================

MODEL_ID = "claude-sonnet-4-6"
MAX_TOKENS = 1024
RAG_TOP_K = 3
MAX_CONTEXT_CHARS = 3000
OFFLINE_REPLY = (
    "I'm running in offline mode — no API key is configured. "
    "Based on the knowledge base, here is what I found:\n\n{context}\n\n"
    "To get full AI-powered responses, add your Anthropic API key in the sidebar."
)

SYSTEM_PROMPT_TEMPLATE = """You are a helpful, friendly customer support assistant.
Answer questions using ONLY the context provided below. If the context does not
contain enough information to answer the question, say so clearly and suggest the
customer contacts support directly.

KNOWLEDGE BASE CONTEXT:
{context}

Guidelines:
- Be concise and direct.
- If referencing a specific document, name it.
- Never fabricate policies, prices, or procedures not in the context.
- If unsure, say "I don't have that information — please contact our support team."
"""


# ============================================================
# BUILT-IN KNOWLEDGE BASE
# ============================================================

KNOWLEDGE_BASE: List[Dict] = [
    {
        "id": "kb-001",
        "title": "Return & Refund Policy",
        "tags": ["return", "refund", "exchange", "policy", "money back"],
        "text": (
            "Items may be returned within 30 days of purchase for a full refund. "
            "Items must be unused and in original packaging. Digital products are "
            "non-refundable once downloaded. To start a return, visit our returns "
            "portal or contact support@example.com with your order number. "
            "Refunds are processed within 5–7 business days."
        ),
    },
    {
        "id": "kb-002",
        "title": "Shipping & Delivery",
        "tags": ["shipping", "delivery", "tracking", "dispatch", "order", "arrive"],
        "text": (
            "Standard shipping takes 3–5 business days. Express shipping (1–2 days) "
            "is available at checkout for an additional fee. Free standard shipping "
            "applies to orders over $50. Orders placed before 2 PM EST are "
            "dispatched the same day. A tracking number is emailed once your order "
            "ships. International shipping is available to 40+ countries."
        ),
    },
    {
        "id": "kb-003",
        "title": "Account & Login Help",
        "tags": ["account", "login", "password", "email", "sign in", "forgot", "reset"],
        "text": (
            "To reset your password, click 'Forgot password' on the login page and "
            "enter your email. A reset link is sent within 2 minutes. If you don't "
            "receive the email, check your spam folder or contact support. You can "
            "update your account email in Settings → Profile. For security, we "
            "recommend enabling two-factor authentication (2FA)."
        ),
    },
    {
        "id": "kb-004",
        "title": "Subscription Plans & Billing",
        "tags": ["subscription", "plan", "billing", "charge", "invoice", "upgrade", "cancel"],
        "text": (
            "We offer three plans: Basic ($9/mo), Pro ($29/mo), and Enterprise "
            "(custom pricing). All plans include a 14-day free trial — no credit "
            "card required. Billing is monthly or annually (annual saves 20%). "
            "You can upgrade, downgrade, or cancel at any time from Settings → "
            "Billing. Cancellations take effect at the end of the current billing "
            "period. Invoices are emailed on the first of each month."
        ),
    },
    {
        "id": "kb-005",
        "title": "Technical Troubleshooting",
        "tags": ["bug", "error", "crash", "not working", "issue", "technical", "support", "problem"],
        "text": (
            "For technical issues: (1) Clear your browser cache and cookies. "
            "(2) Try a different browser or incognito mode. (3) Check our status "
            "page at status.example.com for ongoing incidents. (4) Ensure your "
            "software is up to date. If the problem persists, contact support with "
            "your account email, a description of the issue, and any error messages. "
            "Our technical team responds within 4 business hours."
        ),
    },
    {
        "id": "kb-006",
        "title": "Product Features & Integrations",
        "tags": ["feature", "integration", "api", "connect", "tool", "capability", "how to"],
        "text": (
            "Our platform integrates with Slack, Salesforce, HubSpot, Zapier, and "
            "Google Workspace. API access is available on Pro and Enterprise plans. "
            "The REST API uses OAuth 2.0. Full API docs are at docs.example.com. "
            "Key features include: real-time collaboration, automated workflows, "
            "custom dashboards, and audit logs. Feature requests can be submitted "
            "at feedback.example.com."
        ),
    },
    {
        "id": "kb-007",
        "title": "Privacy & Data Security",
        "tags": ["privacy", "data", "security", "gdpr", "delete", "personal", "safe"],
        "text": (
            "We are GDPR and CCPA compliant. Your data is encrypted in transit "
            "(TLS 1.3) and at rest (AES-256). We never sell personal data to third "
            "parties. You can request a copy of your data or deletion via Settings "
            "→ Privacy, or by emailing privacy@example.com. Data deletion requests "
            "are processed within 30 days. We perform quarterly security audits."
        ),
    },
    {
        "id": "kb-008",
        "title": "Contacting Support",
        "tags": ["contact", "support", "help", "chat", "email", "phone", "human", "agent"],
        "text": (
            "Support channels: Live chat (available 9 AM–6 PM EST, Mon–Fri), "
            "Email (support@example.com, response within 24 hours), and Phone "
            "(1-800-555-0123, available 9 AM–5 PM EST). For urgent issues, use "
            "live chat. For billing, email billing@example.com. Our help centre "
            "at help.example.com contains 200+ self-service articles."
        ),
    },
]


# ============================================================
# RAG CORE
# ============================================================

def score_document(doc: Dict, query: str) -> int:
    """
    Bag-of-words overlap score between a query and a document.

    Checks query tokens against document tags, title, and body text.
    Title/tag matches score double to favour precise results.
    """
    tokens = {t.lower() for t in re.findall(r"\w+", query)}
    tag_title = " ".join(doc["tags"]) + " " + doc["title"].lower()
    body = doc["text"].lower()

    tag_score = sum(2 for t in tokens if t in tag_title)
    body_score = sum(1 for t in tokens if t in body)
    return tag_score + body_score


def rag_retrieve(query: str, corpus: List[Dict], k: int = RAG_TOP_K) -> List[Dict]:
    """Return the top-k most relevant documents for a query."""
    scored = [(score_document(doc, query), doc) for doc in corpus]
    scored = [(s, d) for s, d in scored if s > 0]
    scored.sort(key=lambda x: -x[0])
    results = [d for _, d in scored[:k]]
    return results if results else corpus[:k]


def build_context_block(docs: List[Dict]) -> str:
    """Render retrieved documents into a grounding context string."""
    if not docs:
        return "(No relevant knowledge base articles found.)"
    lines = []
    for i, doc in enumerate(docs, 1):
        lines.append(f"[{i}] {doc['title']}\n{doc['text']}")
    return "\n\n".join(lines)


def truncate_context(context: str, max_chars: int = MAX_CONTEXT_CHARS) -> str:
    """Hard-truncate context to stay within the character budget."""
    if len(context) <= max_chars:
        return context
    return context[:max_chars] + "\n…[truncated]"


def build_system_prompt(docs: List[Dict]) -> str:
    """Build the full system prompt with RAG context injected."""
    context = truncate_context(build_context_block(docs))
    return SYSTEM_PROMPT_TEMPLATE.format(context=context)


# ============================================================
# DOCUMENT MANAGEMENT
# ============================================================

def add_document(corpus: List[Dict], title: str, content: str) -> List[Dict]:
    """
    Add a new document to a corpus copy and return the new list.

    The document id is derived from the current corpus length so it is
    stable within a session.
    """
    new_id = f"user-{len(corpus) + 1:03d}"
    tokens = {t.lower() for t in re.findall(r"\w+", title + " " + content)}
    doc = {
        "id": new_id,
        "title": title.strip(),
        "tags": list(tokens)[:20],
        "text": content.strip(),
    }
    return corpus + [doc]


# ============================================================
# LLM CALL
# ============================================================

def get_client() -> Optional["anthropic.Anthropic"]:
    """Return an Anthropic client if a key is available."""
    if not _ANTHROPIC_IMPORTED:
        return None
    key = os.environ.get("ANTHROPIC_API_KEY") or st.session_state.get("api_key", "")
    if not key:
        return None
    try:
        return anthropic.Anthropic(api_key=key)
    except Exception:
        return None


def format_messages(history: List[Dict]) -> List[Dict]:
    """
    Convert internal chat history format to Anthropic messages format.

    Internal format: [{"role": "user"|"assistant", "content": str}, ...]
    Anthropic format is identical, so this is a pass-through validator.
    """
    return [
        {"role": msg["role"], "content": msg["content"]}
        for msg in history
        if msg.get("role") in {"user", "assistant"} and msg.get("content", "").strip()
    ]


def get_llm_response(
    history: List[Dict],
    system: str,
    client: Optional["anthropic.Anthropic"] = None,
) -> Tuple[str, bool]:
    """
    Call Claude to generate a response.

    Returns (response_text, used_llm). If no client, falls back to an
    offline response that surfaces the RAG context directly.
    """
    if client is None:
        context_start = system.find("KNOWLEDGE BASE CONTEXT:") + len("KNOWLEDGE BASE CONTEXT:")
        context = system[context_start:].strip().split("Guidelines:")[0].strip()
        return OFFLINE_REPLY.format(context=context or "(none)"), False

    msgs = format_messages(history)
    try:
        response = client.messages.create(
            model=MODEL_ID,
            max_tokens=MAX_TOKENS,
            system=system,
            messages=msgs,
        )
        return response.content[0].text, True
    except Exception as exc:
        return f"⚠ LLM error: {exc}\n\nPlease try again or check your API key.", False


# ============================================================
# SESSION STATE
# ============================================================

def init_session_state() -> None:
    if "initialised" in st.session_state:
        return
    st.session_state.messages = []
    st.session_state.corpus = list(KNOWLEDGE_BASE)
    st.session_state.api_key = ""
    st.session_state.conversation_title = "New conversation"
    st.session_state.initialised = True


# ============================================================
# SIDEBAR
# ============================================================

def render_sidebar() -> None:
    with st.sidebar:
        st.markdown("## 💬 Customer Assistant")
        st.caption("RAG-powered support · Powered by Claude")
        st.divider()

        if st.button("+ New conversation", use_container_width=True):
            st.session_state.messages = []
            st.session_state.conversation_title = "New conversation"
            st.rerun()

        st.divider()
        st.markdown("### 🔑 API Key")
        st.text_input(
            "Anthropic API key (session-only)",
            type="password",
            key="api_key",
            label_visibility="collapsed",
            placeholder="sk-ant-...",
        )
        if not _ANTHROPIC_IMPORTED:
            st.info("`anthropic` package not installed — offline mode.")
        elif not (os.environ.get("ANTHROPIC_API_KEY") or st.session_state.get("api_key")):
            st.warning("No key — offline mode active.")
        else:
            st.success("Claude API ready.")

        st.divider()
        st.markdown("### 📚 Knowledge Base")
        st.caption(f"{len(st.session_state.corpus)} articles loaded")
        with st.expander("View articles"):
            for doc in st.session_state.corpus:
                st.markdown(f"- **{doc['title']}** (`{doc['id']}`)")

        st.divider()
        st.markdown("### ➕ Upload document")
        with st.form("upload_doc", clear_on_submit=True):
            doc_title = st.text_input("Document title")
            doc_content = st.text_area("Paste content here", height=120)
            uploaded = st.form_submit_button("Add to knowledge base",
                                             use_container_width=True)
        if uploaded:
            if doc_title.strip() and doc_content.strip():
                st.session_state.corpus = add_document(
                    st.session_state.corpus, doc_title, doc_content
                )
                st.success(f"Added: {doc_title}")
                st.rerun()
            else:
                st.error("Title and content are required.")


# ============================================================
# MAIN CHAT UI
# ============================================================

def render_chat() -> None:
    st.markdown(
        "<h2 style='text-align:center; color:#ececec; margin-bottom:0.25rem;'>"
        "💬 Customer Assistant</h2>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='text-align:center; color:#8e8ea0; margin-top:0;'>"
        "Ask me anything about our products and services.</p>",
        unsafe_allow_html=True,
    )

    # Render conversation history.
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("📖 Sources used", expanded=False):
                    for src in msg["sources"]:
                        st.markdown(f"**{src['title']}** (`{src['id']}`)")
                        st.caption(src["text"][:200] + "…" if len(src["text"]) > 200 else src["text"])

    # Empty state.
    if not st.session_state.messages:
        st.markdown(
            "<div style='text-align:center; margin-top:4rem; color:#8e8ea0;'>"
            "<p style='font-size:1.1rem;'>How can I help you today?</p>"
            "<p style='font-size:0.875rem;'>I can answer questions about returns, "
            "billing, shipping, account issues, and more.</p>"
            "</div>",
            unsafe_allow_html=True,
        )

    # Chat input.
    user_input = st.chat_input("Message Customer Assistant…")
    if not user_input:
        return

    # Retrieve relevant docs.
    sources = rag_retrieve(user_input, st.session_state.corpus)
    system = build_system_prompt(sources)

    # Append user message.
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate response.
    client = get_client()
    with st.chat_message("assistant"):
        with st.spinner(""):
            reply, used_llm = get_llm_response(
                st.session_state.messages[:-1] + [{"role": "user", "content": user_input}],
                system,
                client,
            )
        st.markdown(reply)
        if sources:
            with st.expander("📖 Sources used", expanded=False):
                for src in sources:
                    st.markdown(f"**{src['title']}** (`{src['id']}`)")
                    st.caption(src["text"][:200] + "…" if len(src["text"]) > 200 else src["text"])

    # Save assistant message with source metadata.
    st.session_state.messages.append({
        "role": "assistant",
        "content": reply,
        "sources": sources,
        "used_llm": used_llm,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    })

    # Update conversation title from first user message.
    if len(st.session_state.messages) == 2:
        st.session_state.conversation_title = user_input[:40]


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    init_session_state()
    render_sidebar()
    render_chat()


if __name__ == "__main__":
    main()
