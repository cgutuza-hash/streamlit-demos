"""
Unit tests for customer_assistant/app.py.

All pure-logic functions are tested in isolation. Streamlit is mocked before
import so that module-level st calls (set_page_config, markdown) are safe.
"""

import os
import sys
import pytest
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Mock streamlit BEFORE importing app so module-level st calls are safe.
# ---------------------------------------------------------------------------
_st = MagicMock()
_st.session_state = MagicMock()
_st.session_state.api_key = ""
sys.modules["streamlit"] = _st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import app  # noqa: E402


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

SAMPLE_CORPUS = [
    {
        "id": "kb-001",
        "title": "Return & Refund Policy",
        "tags": ["return", "refund", "exchange", "policy", "money back"],
        "text": "Items may be returned within 30 days for a full refund.",
    },
    {
        "id": "kb-002",
        "title": "Shipping & Delivery",
        "tags": ["shipping", "delivery", "tracking", "order"],
        "text": "Standard shipping takes 3–5 business days.",
    },
    {
        "id": "kb-003",
        "title": "Account & Login Help",
        "tags": ["account", "login", "password", "email", "reset"],
        "text": "To reset your password, click Forgot password on the login page.",
    },
    {
        "id": "kb-004",
        "title": "Subscription Plans & Billing",
        "tags": ["subscription", "plan", "billing", "charge", "cancel"],
        "text": "We offer Basic ($9/mo), Pro ($29/mo), and Enterprise plans.",
    },
]


# ---------------------------------------------------------------------------
# score_document
# ---------------------------------------------------------------------------

class TestScoreDocument:

    def test_exact_tag_match_scores_higher_than_body_match(self):
        doc = {
            "id": "x",
            "title": "Refund Policy",
            "tags": ["refund", "return"],
            "text": "We process refunds within 5 days.",
        }
        tag_score = app.score_document(doc, "refund")
        body_doc = {
            "id": "y",
            "title": "Shipping Info",
            "tags": ["shipping"],
            "text": "Contact us about a refund issue.",
        }
        body_score = app.score_document(body_doc, "refund")
        assert tag_score > body_score

    def test_zero_score_for_unrelated_query(self):
        doc = SAMPLE_CORPUS[0]
        score = app.score_document(doc, "xyzzy nonsense qqq")
        assert score == 0

    def test_score_increases_with_more_matching_tokens(self):
        doc = SAMPLE_CORPUS[0]
        s1 = app.score_document(doc, "refund")
        s2 = app.score_document(doc, "refund return exchange")
        assert s2 > s1

    def test_case_insensitive_matching(self):
        doc = SAMPLE_CORPUS[0]
        s_lower = app.score_document(doc, "refund")
        s_upper = app.score_document(doc, "REFUND")
        assert s_lower == s_upper

    def test_multiple_tokens_scored_independently(self):
        doc = SAMPLE_CORPUS[1]
        s = app.score_document(doc, "shipping tracking")
        assert s > 0


# ---------------------------------------------------------------------------
# rag_retrieve
# ---------------------------------------------------------------------------

class TestRagRetrieve:

    def test_returns_at_most_k_results(self):
        results = app.rag_retrieve("refund return", SAMPLE_CORPUS, k=2)
        assert len(results) <= 2

    def test_returns_most_relevant_first_for_refund(self):
        results = app.rag_retrieve("return refund policy", SAMPLE_CORPUS, k=3)
        assert results[0]["id"] == "kb-001"

    def test_returns_most_relevant_first_for_shipping(self):
        results = app.rag_retrieve("shipping delivery tracking", SAMPLE_CORPUS, k=3)
        assert results[0]["id"] == "kb-002"

    def test_returns_most_relevant_first_for_password(self):
        results = app.rag_retrieve("password reset login account", SAMPLE_CORPUS, k=3)
        assert results[0]["id"] == "kb-003"

    def test_fallback_when_no_match(self):
        results = app.rag_retrieve("xyzzy gibberish qqq", SAMPLE_CORPUS, k=2)
        assert len(results) == 2
        assert results[0]["id"] == SAMPLE_CORPUS[0]["id"]

    def test_result_items_are_dicts_with_required_keys(self):
        results = app.rag_retrieve("billing subscription", SAMPLE_CORPUS, k=2)
        for doc in results:
            assert {"id", "title", "tags", "text"}.issubset(set(doc.keys()))

    def test_k_1_returns_single_result(self):
        results = app.rag_retrieve("shipping", SAMPLE_CORPUS, k=1)
        assert len(results) == 1

    def test_k_larger_than_corpus_returns_all_matches(self):
        results = app.rag_retrieve("refund", SAMPLE_CORPUS, k=100)
        assert len(results) <= len(SAMPLE_CORPUS)


# ---------------------------------------------------------------------------
# build_context_block
# ---------------------------------------------------------------------------

class TestBuildContextBlock:

    def test_empty_docs_returns_fallback(self):
        out = app.build_context_block([])
        assert "No relevant" in out

    def test_includes_all_document_titles(self):
        docs = SAMPLE_CORPUS[:2]
        out = app.build_context_block(docs)
        for doc in docs:
            assert doc["title"] in out

    def test_includes_all_document_texts(self):
        docs = SAMPLE_CORPUS[:2]
        out = app.build_context_block(docs)
        for doc in docs:
            assert doc["text"] in out

    def test_documents_numbered_starting_at_1(self):
        out = app.build_context_block(SAMPLE_CORPUS[:3])
        assert "[1]" in out
        assert "[2]" in out
        assert "[3]" in out

    def test_single_document_context(self):
        out = app.build_context_block([SAMPLE_CORPUS[0]])
        assert "[1]" in out
        assert "[2]" not in out


# ---------------------------------------------------------------------------
# truncate_context
# ---------------------------------------------------------------------------

class TestTruncateContext:

    def test_short_context_unchanged(self):
        ctx = "Hello world"
        assert app.truncate_context(ctx, max_chars=100) == ctx

    def test_long_context_truncated_to_max_chars(self):
        ctx = "x" * 5000
        result = app.truncate_context(ctx, max_chars=100)
        assert len(result) <= 115  # 100 chars + "…[truncated]"

    def test_truncated_ends_with_marker(self):
        ctx = "a" * 1000
        result = app.truncate_context(ctx, max_chars=50)
        assert result.endswith("[truncated]")

    def test_exact_boundary_not_truncated(self):
        ctx = "a" * 100
        assert app.truncate_context(ctx, max_chars=100) == ctx


# ---------------------------------------------------------------------------
# build_system_prompt
# ---------------------------------------------------------------------------

class TestBuildSystemPrompt:

    def test_prompt_contains_context(self):
        docs = [SAMPLE_CORPUS[0]]
        prompt = app.build_system_prompt(docs)
        assert SAMPLE_CORPUS[0]["title"] in prompt

    def test_prompt_contains_guidelines_header(self):
        prompt = app.build_system_prompt(SAMPLE_CORPUS[:2])
        assert "Guidelines" in prompt

    def test_prompt_references_knowledge_base_section(self):
        prompt = app.build_system_prompt(SAMPLE_CORPUS[:1])
        assert "KNOWLEDGE BASE CONTEXT" in prompt

    def test_empty_docs_still_returns_valid_prompt(self):
        prompt = app.build_system_prompt([])
        assert isinstance(prompt, str)
        assert len(prompt) > 0


# ---------------------------------------------------------------------------
# add_document
# ---------------------------------------------------------------------------

class TestAddDocument:

    def test_returns_new_list_with_extra_entry(self):
        new_corpus = app.add_document(SAMPLE_CORPUS, "FAQ", "Frequently asked questions.")
        assert len(new_corpus) == len(SAMPLE_CORPUS) + 1

    def test_original_corpus_not_mutated(self):
        original_len = len(SAMPLE_CORPUS)
        app.add_document(SAMPLE_CORPUS, "FAQ", "Content here.")
        assert len(SAMPLE_CORPUS) == original_len

    def test_new_doc_has_required_keys(self):
        new_corpus = app.add_document(SAMPLE_CORPUS, "My Policy", "Some policy text.")
        doc = new_corpus[-1]
        assert {"id", "title", "tags", "text"}.issubset(set(doc.keys()))

    def test_title_stripped(self):
        new_corpus = app.add_document(SAMPLE_CORPUS, "  Trimmed Title  ", "Content.")
        assert new_corpus[-1]["title"] == "Trimmed Title"

    def test_content_stripped(self):
        new_corpus = app.add_document(SAMPLE_CORPUS, "T", "  Content  ")
        assert new_corpus[-1]["text"] == "Content"

    def test_new_doc_retrievable_by_rag(self):
        new_corpus = app.add_document(
            SAMPLE_CORPUS, "Warranty Policy",
            "All products come with a 2-year warranty. Warranty claims must be filed within 30 days of discovery."
        )
        results = app.rag_retrieve("warranty claim", new_corpus, k=3)
        ids = [d["id"] for d in results]
        assert any("user" in i for i in ids)

    def test_id_is_unique(self):
        corpus1 = app.add_document(SAMPLE_CORPUS, "Doc A", "Content A.")
        corpus2 = app.add_document(corpus1, "Doc B", "Content B.")
        ids = [d["id"] for d in corpus2]
        assert len(ids) == len(set(ids))


# ---------------------------------------------------------------------------
# format_messages
# ---------------------------------------------------------------------------

class TestFormatMessages:

    def test_passes_through_valid_messages(self):
        history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        result = app.format_messages(history)
        assert result == history

    def test_filters_out_empty_content(self):
        history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "   "},
        ]
        result = app.format_messages(history)
        assert len(result) == 1
        assert result[0]["role"] == "user"

    def test_filters_out_invalid_roles(self):
        history = [
            {"role": "user", "content": "Hello"},
            {"role": "system", "content": "You are helpful."},
            {"role": "assistant", "content": "Hi!"},
        ]
        result = app.format_messages(history)
        assert len(result) == 2
        roles = [m["role"] for m in result]
        assert "system" not in roles

    def test_empty_history_returns_empty_list(self):
        assert app.format_messages([]) == []

    def test_preserves_message_order(self):
        history = [
            {"role": "user", "content": "First"},
            {"role": "assistant", "content": "Second"},
            {"role": "user", "content": "Third"},
        ]
        result = app.format_messages(history)
        assert [m["content"] for m in result] == ["First", "Second", "Third"]

    def test_only_role_and_content_fields_in_output(self):
        history = [
            {"role": "user", "content": "Hello", "sources": [], "timestamp": "2026-01-01"},
        ]
        result = app.format_messages(history)
        assert set(result[0].keys()) == {"role", "content"}


# ---------------------------------------------------------------------------
# get_llm_response (offline / no client)
# ---------------------------------------------------------------------------

class TestGetLlmResponseOffline:

    def test_offline_returns_string(self):
        history = [{"role": "user", "content": "What is the return policy?"}]
        system = app.build_system_prompt(SAMPLE_CORPUS[:1])
        reply, used_llm = app.get_llm_response(history, system, client=None)
        assert isinstance(reply, str)
        assert len(reply) > 0

    def test_offline_used_llm_is_false(self):
        history = [{"role": "user", "content": "Hello"}]
        system = app.build_system_prompt([])
        _, used_llm = app.get_llm_response(history, system, client=None)
        assert used_llm is False

    def test_offline_reply_contains_api_key_hint(self):
        history = [{"role": "user", "content": "Help"}]
        system = app.build_system_prompt([])
        reply, _ = app.get_llm_response(history, system, client=None)
        assert "API key" in reply or "offline" in reply.lower()

    def test_llm_error_returns_error_string(self):
        """Simulate a client that raises an exception."""
        bad_client = MagicMock()
        bad_client.messages.create.side_effect = Exception("Connection refused")
        history = [{"role": "user", "content": "Test"}]
        system = app.build_system_prompt([])
        reply, used_llm = app.get_llm_response(history, system, bad_client)
        assert "error" in reply.lower() or "⚠" in reply
        assert used_llm is False

    def test_llm_success_returns_content(self):
        """Simulate a successful LLM response."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Your return window is 30 days.")]
        mock_client.messages.create.return_value = mock_response

        history = [{"role": "user", "content": "How do I return an item?"}]
        system = app.build_system_prompt(SAMPLE_CORPUS[:1])
        reply, used_llm = app.get_llm_response(history, system, mock_client)

        assert reply == "Your return window is 30 days."
        assert used_llm is True

    def test_llm_called_with_correct_model(self):
        """Verify the correct model ID is passed to the API."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Response text")]
        mock_client.messages.create.return_value = mock_response

        history = [{"role": "user", "content": "Test message"}]
        system = "System prompt"
        app.get_llm_response(history, system, mock_client)

        call_kwargs = mock_client.messages.create.call_args
        assert call_kwargs.kwargs.get("model") == app.MODEL_ID

    def test_llm_called_with_system_prompt(self):
        """Verify the system prompt is forwarded to the API."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="OK")]
        mock_client.messages.create.return_value = mock_response

        history = [{"role": "user", "content": "Test"}]
        system = "MY_SYSTEM_PROMPT"
        app.get_llm_response(history, system, mock_client)

        call_kwargs = mock_client.messages.create.call_args
        assert call_kwargs.kwargs.get("system") == "MY_SYSTEM_PROMPT"


# ---------------------------------------------------------------------------
# Integration: end-to-end RAG → prompt → offline response
# ---------------------------------------------------------------------------

class TestEndToEnd:

    def test_refund_query_surfaces_refund_doc(self):
        docs = app.rag_retrieve("return my order refund", SAMPLE_CORPUS, k=3)
        assert any("Refund" in d["title"] for d in docs)

    def test_password_query_surfaces_account_doc(self):
        docs = app.rag_retrieve("forgot my password login", SAMPLE_CORPUS, k=3)
        assert any("Login" in d["title"] or "Account" in d["title"] for d in docs)

    def test_billing_query_surfaces_subscription_doc(self):
        docs = app.rag_retrieve("billing invoice subscription cancel", SAMPLE_CORPUS, k=3)
        assert any("Subscription" in d["title"] or "Billing" in d["title"] for d in docs)

    def test_full_pipeline_returns_string_offline(self):
        query = "How do I cancel my subscription?"
        docs = app.rag_retrieve(query, SAMPLE_CORPUS, k=3)
        system = app.build_system_prompt(docs)
        history = [{"role": "user", "content": query}]
        reply, used_llm = app.get_llm_response(history, system, client=None)
        assert isinstance(reply, str)
        assert len(reply) > 10

    def test_added_document_is_retrieved_by_new_topic(self):
        corpus = app.add_document(
            list(SAMPLE_CORPUS),
            "Pet Insurance FAQ",
            "We cover dogs and cats. Claims must be submitted within 90 days. "
            "Premium is $15/month per pet.",
        )
        docs = app.rag_retrieve("pet insurance claim dog", corpus, k=3)
        assert any("Pet Insurance" in d["title"] for d in docs)

    def test_system_prompt_contains_retrieved_text(self):
        query = "What is the return policy?"
        docs = app.rag_retrieve(query, SAMPLE_CORPUS, k=2)
        prompt = app.build_system_prompt(docs)
        assert "30 days" in prompt or "Return" in prompt
