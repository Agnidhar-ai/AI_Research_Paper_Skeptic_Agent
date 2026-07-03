from datetime import datetime
import html
import importlib
import json
from pathlib import Path
from tempfile import NamedTemporaryFile
import re

import streamlit as st

from agents.skeptic_agent import SkepticAgent
from config import settings
from tools.document_loader import SUPPORTED_DOCUMENT_EXTENSIONS, load_document_text
from tools.limitations import APP_LIMITATIONS
from tools.narrative_report import build_narrative_report, format_narrative_markdown
import tools.paper_chat as paper_chat
import tools.research_workspace as research_workspace
import tools.source_verifier as source_verifier
from tools.report_generator import save_report
from tools.research_workspace import (
    build_knowledge_graph,
    build_learning_flowchart,
    build_section_study_payload,
    build_workspace_sections,
    completed_progress,
    format_knowledge_graph,
    search_workspace,
)
from tools.source_verifier import search_external_support_for_section, summarize_external_verification, verify_external_sources
from tools.text_cleaner import clean_display_text


paper_chat = importlib.reload(paper_chat)
research_workspace = importlib.reload(research_workspace)
source_verifier = importlib.reload(source_verifier)
answer_paper_question = paper_chat.answer_paper_question
build_paper_chat_store = paper_chat.build_paper_chat_store
initialize_discussion_memory = paper_chat.initialize_discussion_memory
skeptic_check_claim = paper_chat.skeptic_check_claim
update_discussion_memory = paper_chat.update_discussion_memory
build_knowledge_graph = research_workspace.build_knowledge_graph
build_learning_flowchart = research_workspace.build_learning_flowchart
build_section_study_payload = research_workspace.build_section_study_payload
build_workspace_sections = research_workspace.build_workspace_sections
completed_progress = research_workspace.completed_progress
format_knowledge_graph = research_workspace.format_knowledge_graph
search_workspace = research_workspace.search_workspace
search_external_support_for_section = source_verifier.search_external_support_for_section
summarize_external_verification = source_verifier.summarize_external_verification
verify_external_sources = source_verifier.verify_external_sources

st.set_page_config(page_title=settings.app_name, layout="wide")
st.title(settings.app_name)
st.caption("Upload a research paper PDF, DOCX, text file, or ZIP, then run a skeptical review for claims, evidence, citations, and reasoning gaps.")

with st.sidebar:
    st.header("Settings")
    st.write(f"LLM provider: `{settings.llm_provider}`")
    st.write(f"Vector store: `{settings.vector_store_backend}`")
    st.write(f"Chunk size: `{settings.chunk_size}`")

UPLOAD_TYPES = sorted(extension.lstrip(".") for extension in SUPPORTED_DOCUMENT_EXTENSIONS)
uploaded_file = st.file_uploader("Upload a research paper PDF, DOCX, text file, or ZIP", type=UPLOAD_TYPES)

if "paper_chat_messages" not in st.session_state:
    st.session_state.paper_chat_messages = []


def clean_ui_text(text: str) -> str:
    text = clean_display_text(str(text))
    replacements = {
        "Â¯": "mean ",
        "¯": "mean ",
        "â€”": "-",
        "â€“": "-",
        "â€™": "'",
        "â€œ": '"',
        "â€": '"',
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    text = re.sub(r"([A-Za-z])-\s+\d{1,4}\s+([A-Za-z])", r"\1\2", text)
    text = re.sub(r"(?<=[A-Za-z])\s+\d{1,4}(?:\s+\d{1,4})+\s+(?=[A-Za-z])", " ", text)
    text = re.sub(r"(?<=[a-z,;:.)])\s+\d{1,4}\s+(?=[A-Za-z])", " ", text)
    text = re.sub(r"(?<=[A-Za-z])\s+\d{1,4}\s+(?=[a-z])", " ", text)
    text = re.sub(r"^\s*\d{1,4}\s+(?=[A-Z][a-z])", "", text)

    compounds = {
        "earlystage": "early-stage",
        "multiturn": "multi-turn",
        "followup": "follow-up",
        "highstakes": "high-stakes",
        "zeroshot": "zero-shot",
        "realworld": "real-world",
    }
    for compact, compound in compounds.items():
        text = re.sub(rf"\b{compact}\b", compound, text, flags=re.IGNORECASE)

    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_snippet_text(text: str, limit: int = 500) -> str:
    text = clean_ui_text(text)
    first_word = text.split(" ", 1)[0] if text else ""
    if first_word and first_word[0].islower() and len(first_word) > 6:
        text = "..." + text
    text = re.sub(r"\bteaching\s+\d{1,4}\b", "teaching", text)
    if text and text[-1] not in ".!?":
        text = text.rsplit(" ", 1)[0].rstrip() + "..."
    if len(text) > limit:
        text = text[:limit].rsplit(" ", 1)[0].rstrip() + "..."
    return text


def evidence_strength(score: float) -> str:
    if score >= 0.45:
        return "strong"
    if score >= 0.25:
        return "moderate"
    return "weak"


def choose_display_match(matches: list[dict]) -> dict:
    if not matches:
        return {}
    return max(matches, key=lambda match: display_match_score(match))


def display_match_score(match: dict) -> float:
    text = clean_ui_text(match.get("text", ""))
    lower_text = text.lower()
    score = float(match.get("score", 0))
    if "anonymous author" in lower_text or "affiliation address email" in lower_text:
        score -= 0.35
    if lower_text.startswith("firm or fickle"):
        score -= 0.25
    if text.startswith("..."):
        score -= 0.05
    if "confidence-aware" in lower_text or "experimental results" in lower_text:
        score += 0.12
    return score


def citation_fallback_label(check: str) -> tuple[str, str]:
    match = re.search(r"(\[[0-9,\-\s]+\])", check)
    citation = match.group(1) if match else "Citation"
    if "," in citation or "-" in citation:
        label = "Grouped citation; verify each referenced work supports the broad claim."
    else:
        label = "Citation support check; verify the cited work directly supports the nearby statement."
    return citation, label


def save_markdown_text(markdown: str, reports_dir: Path, paper_stem: str) -> Path:
    reports_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = reports_dir / f"{paper_stem}_skeptic_report_{timestamp}.md"
    output_path.write_text(markdown, encoding="utf-8")
    return output_path


def render_start_tab(report: dict, narrative: dict, markdown_report: str, uploaded_name: str) -> None:
    st.markdown("### Start Here")
    st.write(narrative["opening"])

    sections = narrative.get("sections", [])
    first_steps = [
        "Read the overview below.",
        "Go to Study to learn one section at a time.",
        "Go to Ask when you want to question the paper.",
        "Go to Verify when you want external context or citation checks.",
    ]
    for step in first_steps:
        st.write(f"- {step}")

    if sections:
        st.markdown("### Key Sections")
        for section in sections[:4]:
            st.write(f"**{section['title']}**")
            st.caption(clean_snippet_text(section["body"], limit=220))

    render_limitations_panel()

    st.download_button(
        "Download Markdown report",
        markdown_report,
        file_name=f"{Path(uploaded_name).stem}_skeptic_report.md",
        mime="text/markdown",
    )


def render_limitations_panel(expanded: bool = False) -> None:
    with st.expander("Known limitations", expanded=expanded):
        st.write("These points keep the review honest and prevent over-trusting the output.")
        for limitation in APP_LIMITATIONS:
            st.write(f"- {limitation}")


def render_external_status(payload: dict) -> None:
    summary = payload.get("status_summary") or summarize_external_verification(
        payload.get("status", "no_results"),
        int(payload.get("result_count", 0)),
        payload.get("errors", []),
    )
    message = f"**{summary['label']}**: {summary['message']}"
    tone = summary.get("tone")
    if tone == "success":
        st.success(message)
    elif tone == "error":
        st.error(message)
    else:
        st.warning(message)


def render_flowchart_tab(report: dict, workspace_sections: list[dict]) -> None:
    st.markdown("### Visual Flowchart")
    nodes = build_learning_flowchart(report, workspace_sections)
    if not nodes:
        st.caption("No flowchart could be built from this review yet.")
        return

    html_parts = [
        """
        <style>
        .flowchart-wrap {
            display: flex;
            flex-wrap: wrap;
            align-items: stretch;
            gap: 14px;
            margin: 12px 0 22px;
        }
        .flow-node {
            flex: 1 1 210px;
            min-width: 190px;
            border-radius: 8px;
            padding: 15px 16px;
            color: #ffffff;
            box-shadow: 0 12px 28px rgba(15, 23, 42, 0.20);
            border: 1px solid rgba(255, 255, 255, 0.20);
            position: relative;
        }
        .flow-node h4 {
            margin: 0 0 8px;
            color: #ffffff;
            font-size: 1.02rem;
            line-height: 1.25;
        }
        .flow-node p {
            margin: 0;
            color: rgba(255, 255, 255, 0.92);
            font-size: 0.90rem;
            line-height: 1.35;
        }
        .flow-number {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 28px;
            height: 28px;
            border-radius: 50%;
            background: rgba(255, 255, 255, 0.22);
            font-weight: 700;
            margin-bottom: 10px;
        }
        .flow-button {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            margin-top: 14px;
            padding: 8px 12px;
            border-radius: 7px;
            background: rgba(255, 255, 255, 0.96);
            color: #0f172a !important;
            font-weight: 700;
            text-decoration: none !important;
            font-size: 0.88rem;
        }
        .flow-button:hover {
            background: #ffffff;
            box-shadow: 0 8px 20px rgba(15, 23, 42, 0.22);
        }
        .flow-arrow {
            align-self: center;
            color: transparent;
            font-weight: 800;
            font-size: 0;
            line-height: 1;
        }
        .flow-arrow::before {
            content: "\\2192";
            color: #64748b;
            font-size: 1.6rem;
        }
        .flow-anchor-space {
            min-height: 55vh;
        }
        @media (max-width: 760px) {
            .flowchart-wrap {
                flex-direction: column;
            }
            .flow-arrow {
                transform: rotate(90deg);
                align-self: flex-start;
                margin-left: 18px;
            }
        }
        </style>
        <div class="flowchart-wrap">
        """
    ]
    for index, node in enumerate(nodes, start=1):
        color = html.escape(str(node.get("color", "#2563eb")))
        title = html.escape(str(node.get("title", f"Step {index}")))
        body = html.escape(str(node.get("body", "")))
        topic_id = html.escape(str(node.get("topic_id", f"topic-{index}")))
        html_parts.append(
            f"""
            <div class="flow-node" style="background: linear-gradient(135deg, {color}, #0f172a);">
                <div class="flow-number">{index}</div>
                <h4>{title}</h4>
                <p>{body}</p>
                <a class="flow-button" href="#flow-topic-{topic_id}">Open topic</a>
            </div>
            """
        )
        if index < len(nodes):
            html_parts.append('<div class="flow-arrow">→</div>')
    html_parts.append("</div>")
    quick_path = " &rarr; ".join(html.escape(str(node.get("title", "Step"))) for node in nodes)
    html_parts.append(
        f"""
        <h3 style="margin: 22px 0 10px; font-family: Segoe UI, Arial, sans-serif;">Quick Path</h3>
        <div style="
            padding: 12px 14px;
            border-radius: 8px;
            background: rgba(148, 163, 184, 0.12);
            color: #334155;
            font-family: Segoe UI, Arial, sans-serif;
            font-weight: 700;
            line-height: 1.45;
        ">{quick_path}</div>
        <h3 style="margin: 22px 0 10px; font-family: Segoe UI, Arial, sans-serif;">Topic Landing Points</h3>
        """
    )
    for topic_index, topic_node in enumerate(nodes, start=1):
        topic_id = html.escape(str(topic_node.get("topic_id", f"topic-{topic_index}")))
        topic_title = html.escape(str(topic_node.get("topic_title", topic_node.get("title", f"Topic {topic_index}"))))
        topic_body = html.escape(str(topic_node.get("topic_body", topic_node.get("body", ""))))
        topic_keywords = html.escape(", ".join(str(keyword) for keyword in topic_node.get("keywords", []) if keyword))
        topic_color = html.escape(str(topic_node.get("color", "#2563eb")))
        html_parts.append(
            f"""
            <div id="flow-topic-{topic_id}" style="
                border-left: 7px solid {topic_color};
                padding: 14px 16px;
                margin: 14px 0;
                border-radius: 8px;
                background: rgba(148, 163, 184, 0.10);
                font-family: Segoe UI, Arial, sans-serif;
            ">
                <h4 style="margin: 0 0 8px; color: #0f172a;">{topic_index}. {topic_title}</h4>
                <p style="margin: 0 0 8px; color: #334155; line-height: 1.45;">{topic_body}</p>
                <p style="margin: 0; color: #64748b;"><strong>Related:</strong> {topic_keywords}</p>
            </div>
            """
        )
    html_parts.append('<div class="flow-anchor-space"></div>')
    st.html("".join(html_parts))
    return

    st.markdown("#### Quick Path")
    st.write(" → ".join(node.get("title", "Step") for node in nodes))

    st.markdown("#### Topic Landing Points")
    for index, node in enumerate(nodes, start=1):
        topic_id = html.escape(str(node.get("topic_id", f"topic-{index}")))
        title = html.escape(str(node.get("topic_title", node.get("title", f"Topic {index}"))))
        topic_body = html.escape(str(node.get("topic_body", node.get("body", ""))))
        keywords = [str(keyword) for keyword in node.get("keywords", []) if keyword]
        keyword_text = html.escape(", ".join(keywords))
        color = html.escape(str(node.get("color", "#2563eb")))
        st.markdown(
            f"""
            <div id="flow-topic-{topic_id}" style="
                border-left: 7px solid {color};
                padding: 14px 16px;
                margin: 14px 0;
                border-radius: 8px;
                background: rgba(148, 163, 184, 0.10);
            ">
                <h4 style="margin: 0 0 8px;">{index}. {title}</h4>
                <p style="margin: 0 0 8px;">{topic_body}</p>
                <p style="margin: 0; color: #64748b;"><strong>Related:</strong> {keyword_text}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_study_tab(report: dict, workspace_sections: list[dict]) -> None:
    st.markdown("### Study One Section")
    st.caption("Pick one section, choose your level, and read the explanation. Advanced details stay folded away.")

    workspace_notes = st.session_state.get("workspace_notes", {})
    completed_sections = st.session_state.get("completed_workspace_sections", [])
    progress = completed_progress(completed_sections, workspace_sections)

    progress_cols = st.columns([1, 2])
    progress_cols[0].progress(progress["ratio"])
    progress_cols[1].caption(f"{progress['completed']} of {progress['total']} sections completed")

    section_titles = [section["title"] for section in workspace_sections]
    if "workspace_selected_section" not in st.session_state and section_titles:
        st.session_state.workspace_selected_section = section_titles[0]

    control_cols = st.columns([1.6, 1.2, 1.2])
    selected_title = control_cols[0].selectbox(
        "Paper map",
        section_titles,
        key="workspace_selected_section",
    )
    reading_mode = control_cols[1].selectbox("Mode", ["Explain", "Evidence", "Skeptic"])
    understanding_level = control_cols[2].selectbox(
        "Level",
        ["Beginner", "Intermediate", "Researcher", "Explain Like 7 Years Old", "Exam Preparation"],
    )

    selected_section = next(
        (section for section in workspace_sections if section["title"] == selected_title),
        workspace_sections[0],
    )
    study_payload = build_section_study_payload(selected_section, understanding_level, reading_mode, report)

    st.markdown(f"### {selected_section['title']}")
    st.caption(f"Source: {selected_section.get('source', 'Paper')} | Confidence: {study_payload['confidence']['level']}")

    if reading_mode == "Evidence":
        st.markdown("#### Paper Evidence")
        st.write(study_payload["paper_content"])
        st.markdown("#### Verification Questions With Answers")
        for item in study_payload["verification_answers"]:
            st.write(f"**Question:** {item['question']}")
            st.write(f"**Answer:** {item['answer']}")
            st.divider()

        st.markdown("#### External Support")
        st.caption("Optional: search Wikipedia and arXiv for sources related to this section. This is context, not proof.")
        support_key = f"external_support_{selected_section['title']}"
        if "section_external_support" not in st.session_state:
            st.session_state.section_external_support = {}
        if st.button("Find external support for this section"):
            with st.spinner("Searching external sources"):
                st.session_state.section_external_support[support_key] = search_external_support_for_section(selected_section)
            st.success("External support search finished. See results below.")

        external_support = st.session_state.section_external_support.get(support_key)
        if external_support:
            st.info(external_support["disclaimer"])
            st.write(f"Search query: `{external_support['query']}`")
            render_external_status(external_support)
            for error in external_support.get("errors", []):
                st.warning(f"Search issue: {error}")
            for source_name in ("wikipedia", "arxiv"):
                results = external_support.get(source_name, [])
                with st.expander(f"{source_name.title()} support", expanded=source_name == "wikipedia"):
                    if not results:
                        st.caption(f"No {source_name.title()} support returned.")
                    for result in results[:4]:
                        if result.get("error"):
                            st.warning(f"{result.get('query', external_support['query'])}: {result['error']}")
                            continue
                        st.markdown(f"**[{clean_ui_text(result.get('title', 'Untitled source'))}]({result.get('url', '')})**")
                        st.caption(clean_snippet_text(result.get("snippet", ""), limit=320))
    elif reading_mode == "Skeptic":
        st.markdown("#### Skeptical Analysis")
        for question_item in study_payload["skeptical_questions"]:
            st.write(f"- {question_item}")
        with st.expander("Paper content"):
            st.write(study_payload["paper_content"])
    else:
        st.markdown("#### Simple Explanation")
        st.write(study_payload["simple_explanation"])
        st.markdown("#### Real-World Example")
        st.write(study_payload["real_world_example"])
        st.markdown("#### Why It Matters")
        st.write(study_payload["why_it_matters"])
        st.markdown("#### Related Concepts")
        st.write(", ".join(study_payload["related_concepts"]) or "No related concepts detected yet.")

    action_cols = st.columns([1, 1])
    if action_cols[0].button("Mark this section complete"):
        completed = list(st.session_state.get("completed_workspace_sections", []))
        if selected_title not in completed:
            completed.append(selected_title)
        st.session_state.completed_workspace_sections = completed
        st.rerun()

    with st.expander("My note for this section"):
        note_text = st.text_area(
            "Note",
            value=workspace_notes.get(selected_section["title"], ""),
            key=f"note_{selected_section['title']}",
            label_visibility="collapsed",
            height=120,
        )
        workspace_notes[selected_section["title"]] = note_text
        st.session_state.workspace_notes = workspace_notes

    with st.expander("Search inside workspace"):
        workspace_query = st.text_input("Search paper, concepts, and notes")
        if workspace_query:
            results = search_workspace(workspace_sections, workspace_query, workspace_notes)
            if results:
                for result in results:
                    st.write(f"**{result['title']}** - {result['source']}")
                    st.caption(clean_snippet_text(result["snippet"], limit=260))
            else:
                st.caption("No matches found.")


def render_ask_tab(report: dict) -> None:
    st.markdown("### Ask the Paper")
    st.caption("Use the buttons for common questions. For exact claims, keep questions specific.")

    chat_mode = st.radio(
        "Mode",
        ["Ask questions", "Skeptic check"],
        horizontal=True,
        help="Use Skeptic check to audit a claim or proposed answer against retrieved paper evidence.",
    )

    web_search_enabled = st.checkbox(
        "Use web search only when paper evidence is weak",
        value=False,
        help="When off, answers stay inside the paper and final review.",
        disabled=chat_mode == "Skeptic check",
    )

    if chat_mode == "Skeptic check":
        st.markdown("#### Evidence-check a claim")
        claim = st.text_area(
            "Claim or proposed answer",
            placeholder="Paste a claim like: The method significantly improves multi-turn consistency.",
            height=120,
        )
        if st.button("Run skeptic check"):
            response = skeptic_check_claim(claim, st.session_state.get("paper_chat_store"), report=report)
            st.write(response["answer"])
            st.caption(
                f"Verdict: {response['verdict']} | Confidence: {response['confidence']} - "
                f"{response['confidence_reason']}"
            )
            if response.get("follow_up_questions"):
                st.markdown("#### Follow-up questions")
                for follow_up in response["follow_up_questions"]:
                    st.write(f"- {follow_up}")
            for evidence in response.get("evidence", [])[:3]:
                st.caption(f"Evidence ({evidence['score']:.3f}): {clean_snippet_text(evidence['text'], limit=260)}")
        return

    suggested_questions = [
        ("Reading guide", "how should i go through it"),
        ("Overview", "what does this paper say"),
        ("Contributions", "what are the technical contributions"),
        ("Weaknesses", "what are the weaknesses"),
    ]
    suggestion_columns = st.columns(len(suggested_questions))
    for index, (label, suggested_question) in enumerate(suggested_questions):
        if suggestion_columns[index].button(label, key=f"suggested_question_{index}"):
            st.session_state.pending_chat_question = suggested_question

    if st.button("Clear chat"):
        st.session_state.paper_chat_messages = []
        st.session_state.discussion_memory = initialize_discussion_memory(report)
        st.session_state.pending_chat_question = ""
        st.rerun()

    discussion_memory = st.session_state.get("discussion_memory") or initialize_discussion_memory(report)
    with st.expander("Discussion memory"):
        st.write(f"Current section: {discussion_memory.get('current_section') or 'Not set yet'}")
        st.write(f"Covered: {', '.join(discussion_memory.get('covered', [])) or 'None yet'}")
        st.write(f"Remaining: {', '.join(discussion_memory.get('remaining', [])) or 'None'}")

    for message in st.session_state.paper_chat_messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if message["role"] == "assistant":
                fallback_source = {
                    "report": "Final review",
                    "heuristic": "Paper evidence search",
                    "guarded": "Paper evidence search",
                    "web": "Web plus paper evidence",
                    "web_error": "Web search error",
                }.get(message.get("mode"), "Paper chat")
                sources = message.get("sources") or [fallback_source]
                if message.get("confidence"):
                    confidence_text = f"Confidence: {message['confidence']}"
                    if message.get("confidence_reason"):
                        confidence_text += f" - {message['confidence_reason']}"
                    st.caption(confidence_text)
                st.caption(f"Answer sources: {', '.join(sources)}")
            for evidence in message.get("evidence", [])[:2]:
                st.caption(f"Evidence ({evidence['score']:.3f}): {clean_snippet_text(evidence['text'], limit=260)}")
            for web_result in message.get("web_results", [])[:2]:
                if web_result.get("error"):
                    st.caption(f"Web error: {web_result.get('source', 'Web')} - {web_result['error']}")
                else:
                    st.caption(
                        f"Web: {clean_ui_text(web_result.get('title', 'Untitled source'))} "
                        f"{web_result.get('url', '')}"
                    )
            for web_error in message.get("web_errors", [])[:2]:
                st.caption(f"Web search status: {web_error}")

    question = st.session_state.get("pending_chat_question") or st.chat_input("Ask the paper a question")
    st.session_state.pending_chat_question = ""
    if question:
        st.session_state.paper_chat_messages.append({"role": "user", "content": question})
        response = answer_paper_question(
            question,
            st.session_state.get("paper_chat_store"),
            st.session_state.get("paper_report"),
            discussion_memory=st.session_state.get("discussion_memory"),
            web_search_enabled=web_search_enabled,
        )
        st.session_state.discussion_memory = update_discussion_memory(
            st.session_state.get("discussion_memory"),
            question,
            response,
            st.session_state.get("paper_report"),
        )
        st.session_state.paper_chat_messages.append(
            {
                "role": "assistant",
                "content": response["answer"],
                "evidence": response.get("evidence", []),
                "web_results": response.get("web_results", []),
                "web_errors": response.get("web_errors", []),
                "mode": response.get("mode"),
                "confidence": response.get("confidence"),
                "confidence_reason": response.get("confidence_reason"),
                "sources": response.get("sources", []),
            }
        )
        st.rerun()


def render_verify_tab(report: dict) -> None:
    st.markdown("### Verify")
    st.caption("Use this when you want external context or citation/evidence checks. External results are not proof.")

    if st.button("Check Wikipedia and arXiv"):
        with st.spinner("Checking external sources"):
            st.session_state.external_sources = verify_external_sources(report)

    external_sources = st.session_state.get("external_sources")
    if external_sources:
        st.info(external_sources["disclaimer"])
        st.write(f"Queries: {', '.join(external_sources.get('queries', []))}")
        render_external_status(external_sources)
        for error in external_sources.get("errors", []):
            st.warning(f"Search issue: {error}")

        for source_name in ("wikipedia", "arxiv"):
            results = external_sources.get(source_name, [])
            with st.expander(f"{source_name.title()} results", expanded=source_name == "wikipedia"):
                if not results:
                    st.caption(f"No {source_name.title()} results returned.")
                for result in results[:6]:
                    if result.get("error"):
                        st.warning(f"{result['query']}: {result['error']}")
                        continue
                    st.markdown(f"**[{clean_ui_text(result['title'])}]({result.get('url', '')})**")
                    st.caption(clean_snippet_text(result.get("snippet", ""), limit=300))

    with st.expander("Citation checks"):
        citation_details = report.get("citation_details", [])
        if citation_details:
            grouped = [item["citation"] for item in citation_details if item.get("count", 0) >= 3]
            claim_bearing = [
                item["citation"]
                for item in citation_details
                if "Claim-bearing" in item.get("check", "")
            ]
            st.write(f"Grouped citations: {', '.join(grouped[:8]) or 'none detected'}")
            st.write(f"Claim-bearing citations: {', '.join(claim_bearing[:8]) or 'none detected'}")
        else:
            for check in report.get("citation_checks", [])[:10]:
                citation, label = citation_fallback_label(clean_ui_text(check))
                st.write(f"- **{citation}**: {label}")


def render_advanced_tab(report: dict, narrative: dict) -> None:
    st.markdown("### Advanced")
    st.caption("Raw review details for debugging, grading, or deeper inspection.")
    render_limitations_panel()

    with st.expander("Final review", expanded=True):
        st.write(narrative["opening"])
        for section in narrative["sections"]:
            st.markdown(f"### {section['title']}")
            st.write(section["body"])

    with st.expander("Skeptical findings"):
        for finding in report["skeptical_findings"]:
            if isinstance(finding, dict):
                severity = finding.get("severity", "note").upper()
                st.write(f"- **{severity}:** {clean_ui_text(finding.get('finding', finding))}")
            else:
                st.write(f"- {clean_ui_text(finding)}")

    with st.expander("Claims reviewed"):
        for claim in report.get("claims", [])[:8]:
            st.write(
                f"- **{claim.get('risk_level', 'unknown').upper()} risk:** "
                f"{clean_ui_text(claim.get('claim', claim))}"
            )

    with st.expander("Evidence readout"):
        for item in report.get("evidence", [])[:3]:
            matches = item.get("matches", [])
            if matches:
                display_match = choose_display_match(matches)
                score = float(display_match.get("score", 0))
                strength = evidence_strength(score)
                st.write(f"**Claim:** {clean_ui_text(item.get('claim', ''))}")
                st.write(f"Evidence signal: **{strength}** (`{score:.3f}`)")
                st.caption(clean_snippet_text(str(display_match.get("text", "")), limit=360))

    with st.expander("Knowledge graph"):
        st.code(format_knowledge_graph(build_knowledge_graph(report)), language="text")

    with st.expander("Structured JSON report"):
        st.json(report)


def latest_saved_report_path() -> Path | None:
    reports = saved_report_paths()
    return reports[0] if reports else None


def saved_report_paths(limit: int = 30) -> list[Path]:
    reports_dir = Path(settings.outputs_dir) / "reports"
    if not reports_dir.exists():
        return []
    reports = sorted(reports_dir.glob("*_skeptic_report_*.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    return reports[:limit]


def saved_report_label(report_path: Path) -> str:
    source = report_path.name
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        source = clean_ui_text(report.get("source") or source)
    except (OSError, json.JSONDecodeError):
        source = report_path.name
    modified = datetime.fromtimestamp(report_path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
    return f"{source} | {modified} | {report_path.name}"


def render_saved_report_selector(saved_reports: list[Path]) -> None:
    if not saved_reports:
        return

    st.subheader("Restore a saved review")
    st.caption("Choose the exact saved report you want. This avoids accidentally opening the wrong paper.")
    labels = [saved_report_label(path) for path in saved_reports]
    selected_label = st.selectbox("Saved reports", labels, key="saved_report_selector")
    selected_path = saved_reports[labels.index(selected_label)]
    if st.button("Restore selected saved review"):
        restore_saved_review(selected_path)
        st.rerun()


def restore_saved_review(report_path: Path) -> None:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if "narrative_review" not in report:
        report["narrative_review"] = build_narrative_report(report)
    st.session_state.paper_text = ""
    st.session_state.paper_report = report
    st.session_state.paper_chat_store = None
    st.session_state.workspace_sections = build_workspace_sections("", report)
    st.session_state.workspace_notes = {}
    st.session_state.completed_workspace_sections = []
    st.session_state.paper_chat_messages = []
    st.session_state.discussion_memory = initialize_discussion_memory(report)
    st.session_state.external_sources = None
    st.session_state.section_external_support = {}
    markdown_path = report_path.with_suffix(".md")
    st.session_state.report_paths = (str(report_path), str(markdown_path) if markdown_path.exists() else "")
    st.session_state.restored_report_name = report.get("source") or report_path.name


def render_guided_workspace(report: dict, source_name: str) -> None:
    output_path, markdown_path = st.session_state.get("report_paths", ("", ""))
    if output_path and markdown_path:
        st.success(f"Reports saved to {output_path} and {markdown_path}")
    elif output_path:
        st.success(f"Report restored from {output_path}")

    narrative = report.get("narrative_review") or build_narrative_report(report)
    markdown_report = format_narrative_markdown(report)
    workspace_sections = st.session_state.get("workspace_sections") or build_workspace_sections(
        st.session_state.get("paper_text", ""),
        report,
    )
    st.session_state.workspace_sections = workspace_sections

    st.subheader("Guided Research Workspace")
    st.caption("Use the tabs from left to right: Start, Flowchart, Study, Ask, Verify, then Advanced only if needed.")

    total_sections = len(workspace_sections)
    completed_sections = len(st.session_state.get("completed_workspace_sections", []))
    status_cols = st.columns(3)
    status_cols[0].metric("Sections", total_sections)
    status_cols[1].metric("Completed", completed_sections)
    status_cols[2].metric("Chat turns", len(st.session_state.get("paper_chat_messages", [])) // 2)

    start_tab, flowchart_tab, study_tab, ask_tab, verify_tab, advanced_tab = st.tabs(
        ["1 Start", "2 Flowchart", "3 Study", "4 Ask", "5 Verify", "Advanced"]
    )
    with start_tab:
        render_start_tab(report, narrative, markdown_report, source_name)
    with flowchart_tab:
        render_flowchart_tab(report, workspace_sections)
    with study_tab:
        render_study_tab(report, workspace_sections)
    with ask_tab:
        render_ask_tab(report)
    with verify_tab:
        render_verify_tab(report)
    with advanced_tab:
        render_advanced_tab(report, narrative)


if uploaded_file is None:
    restored_report = st.session_state.get("paper_report")
    saved_reports = saved_report_paths()

    if restored_report:
        st.info("Restored a saved review. Upload a new document above when you want to start a new review.")
        render_saved_report_selector(saved_reports)
        render_guided_workspace(restored_report, st.session_state.get("restored_report_name", "Restored review"))
    else:
        st.info("Choose a document to begin. After upload, a **Review paper** button will appear here.")

        st.subheader("What the review checks")
        col1, col2, col3 = st.columns(3)
        col1.metric("Claims", "Extract")
        col2.metric("Evidence", "Retrieve")
        col3.metric("Citations", "Check")

        st.write(
            "The app will parse the document, split it into searchable chunks, identify likely claims or topics, "
            "look for supporting context, and save a structured report in `outputs/reports`."
        )

        render_limitations_panel()
        render_saved_report_selector(saved_reports)
else:
    st.success(f"Uploaded: {uploaded_file.name}")

    with NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as temp_file:
        temp_file.write(uploaded_file.getbuffer())
        temp_path = Path(temp_file.name)

    if st.button("Review paper", type="primary"):
        with st.spinner("Reviewing the paper"):
            paper_text = load_document_text(temp_path)
            report = SkepticAgent().review(paper_text, source_name=uploaded_file.name)
            report["narrative_review"] = build_narrative_report(report)
            chat_store = build_paper_chat_store(paper_text, uploaded_file.name)
            output_path = save_report(
                report,
                Path(settings.outputs_dir) / "reports",
                Path(uploaded_file.name).stem,
            )
            markdown_report = format_narrative_markdown(report)
            markdown_path = save_markdown_text(
                markdown_report,
                Path(settings.outputs_dir) / "reports",
                Path(uploaded_file.name).stem,
            )
            st.session_state.paper_text = paper_text
            st.session_state.paper_report = report
            st.session_state.paper_chat_store = chat_store
            st.session_state.workspace_sections = build_workspace_sections(paper_text, report)
            st.session_state.workspace_notes = {}
            st.session_state.completed_workspace_sections = []
            st.session_state.paper_chat_messages = []
            st.session_state.discussion_memory = initialize_discussion_memory(report)
            st.session_state.external_sources = None
            st.session_state.section_external_support = {}
            st.session_state.report_paths = (output_path, markdown_path)

    report = st.session_state.get("paper_report")
    if report:
        output_path, markdown_path = st.session_state.get("report_paths", ("", ""))
        if output_path and markdown_path:
            st.success(f"Reports saved to {output_path} and {markdown_path}")

        narrative = report.get("narrative_review") or build_narrative_report(report)
        markdown_report = format_narrative_markdown(report)
        workspace_sections = st.session_state.get("workspace_sections") or build_workspace_sections(
            st.session_state.get("paper_text", ""),
            report,
        )
        st.session_state.workspace_sections = workspace_sections

        st.subheader("Guided Research Workspace")
        st.caption("Use the tabs from left to right: Start, Flowchart, Study, Ask, Verify, then Advanced only if needed.")

        total_sections = len(workspace_sections)
        completed_sections = len(st.session_state.get("completed_workspace_sections", []))
        status_cols = st.columns(3)
        status_cols[0].metric("Sections", total_sections)
        status_cols[1].metric("Completed", completed_sections)
        status_cols[2].metric("Chat turns", len(st.session_state.get("paper_chat_messages", [])) // 2)

        start_tab, flowchart_tab, study_tab, ask_tab, verify_tab, advanced_tab = st.tabs(
            ["1 Start", "2 Flowchart", "3 Study", "4 Ask", "5 Verify", "Advanced"]
        )
        with start_tab:
            render_start_tab(report, narrative, markdown_report, uploaded_file.name)
        with flowchart_tab:
            render_flowchart_tab(report, workspace_sections)
        with study_tab:
            render_study_tab(report, workspace_sections)
        with ask_tab:
            render_ask_tab(report)
        with verify_tab:
            render_verify_tab(report)
        with advanced_tab:
            render_advanced_tab(report, narrative)

        st.stop()

        output_path, markdown_path = st.session_state.get("report_paths", ("", ""))
        if output_path and markdown_path:
            st.success(f"Reports saved to {output_path} and {markdown_path}")

        narrative = report.get("narrative_review") or build_narrative_report(report)
        markdown_report = format_narrative_markdown(report)

        workspace_sections = st.session_state.get("workspace_sections") or build_workspace_sections(
            st.session_state.get("paper_text", ""),
            report,
        )
        st.session_state.workspace_sections = workspace_sections

        st.subheader("Research Workspace")
        toolbar_cols = st.columns([1.2, 1.2, 1, 1.6])
        reading_mode = toolbar_cols[0].selectbox("Reading mode", ["Explain", "Evidence", "Skeptic"])
        understanding_level = toolbar_cols[1].selectbox(
            "Understanding level",
            ["Beginner", "Intermediate", "Researcher", "Explain Like 7 Years Old", "Exam Preparation"],
        )
        workspace_web_search = toolbar_cols[2].checkbox("Web search", value=False)
        workspace_query = toolbar_cols[3].text_input("Search paper, concepts, notes")

        workspace_notes = st.session_state.get("workspace_notes", {})
        if workspace_query:
            results = search_workspace(workspace_sections, workspace_query, workspace_notes)
            with st.expander("Search results", expanded=True):
                if results:
                    for result in results:
                        st.write(f"**{result['title']}** · {result['source']}")
                        st.caption(clean_snippet_text(result["snippet"], limit=260))
                else:
                    st.caption("No workspace matches found.")

        completed_sections = st.session_state.get("completed_workspace_sections", [])
        progress = completed_progress(completed_sections, workspace_sections)

        left_panel, main_panel, right_panel = st.columns([1.05, 2.45, 1.25])
        section_titles = [section["title"] for section in workspace_sections]
        if "workspace_selected_section" not in st.session_state and section_titles:
            st.session_state.workspace_selected_section = section_titles[0]

        with left_panel:
            st.markdown("### Paper Map")
            selected_title = st.radio(
                "Sections",
                section_titles,
                key="workspace_selected_section",
                label_visibility="collapsed",
            )
            st.progress(progress["ratio"])
            st.caption(f"{progress['completed']} of {progress['total']} sections completed")
            if st.button("Mark section complete"):
                completed = list(st.session_state.get("completed_workspace_sections", []))
                if selected_title not in completed:
                    completed.append(selected_title)
                st.session_state.completed_workspace_sections = completed
                st.rerun()

        selected_section = next(
            (section for section in workspace_sections if section["title"] == st.session_state.workspace_selected_section),
            workspace_sections[0],
        )
        study_payload = build_section_study_payload(selected_section, understanding_level, reading_mode, report)

        with main_panel:
            st.markdown(f"### {selected_section['title']}")
            st.caption(f"Source: {selected_section.get('source', 'Paper')}")

            if reading_mode == "Evidence":
                st.markdown("#### Paper Evidence")
                st.write(study_payload["paper_content"])
                st.markdown("#### Verification Questions")
                for question_item in study_payload["skeptical_questions"]:
                    st.write(f"- {question_item}")
            elif reading_mode == "Skeptic":
                st.markdown("#### Skeptical Analysis")
                for question_item in study_payload["skeptical_questions"]:
                    st.write(f"- {question_item}")
                st.markdown("#### Paper Content")
                st.write(study_payload["paper_content"])
            else:
                st.markdown("#### Simple Explanation")
                st.write(study_payload["simple_explanation"])
                st.markdown("#### Real-World Example")
                st.write(study_payload["real_world_example"])
                st.markdown("#### Why It Matters")
                st.write(study_payload["why_it_matters"])
                st.markdown("#### Related Concepts")
                st.write(", ".join(study_payload["related_concepts"]) or "No related concepts detected yet.")
                with st.expander("Paper content"):
                    st.write(study_payload["paper_content"])

        with right_panel:
            st.markdown("### Research Tools")
            st.write(f"Confidence: **{study_payload['confidence']['level']}**")
            st.caption(study_payload["confidence"]["reason"])
            st.write(f"Web search: **{'ON' if workspace_web_search else 'OFF'}**")

            memory = st.session_state.get("discussion_memory") or initialize_discussion_memory(report)
            with st.expander("Discussion Memory", expanded=True):
                st.caption(f"Current: {memory.get('current_section') or selected_section['title']}")
                st.caption(f"Covered: {', '.join(memory.get('covered', [])) or 'None yet'}")
                st.caption(f"Remaining: {', '.join(memory.get('remaining', [])) or 'None'}")

            note_key = f"note_{selected_section['title']}"
            note_text = st.text_area(
                "My Notes",
                value=workspace_notes.get(selected_section["title"], ""),
                key=note_key,
                height=130,
            )
            workspace_notes[selected_section["title"]] = note_text
            st.session_state.workspace_notes = workspace_notes

            with st.expander("Knowledge Graph", expanded=False):
                st.code(format_knowledge_graph(build_knowledge_graph(report)), language="text")

            st.write("Paper citation")
            st.caption(report.get("source", uploaded_file.name))

        with st.expander("Final Review"):
            st.write(narrative["opening"])
            for section in narrative["sections"]:
                st.markdown(f"### {section['title']}")
                st.write(section["body"])

        st.download_button(
            "Download Markdown report",
            markdown_report,
            file_name=f"{Path(uploaded_file.name).stem}_skeptic_report.md",
            mime="text/markdown",
        )

        st.subheader("External Source Check")
        st.caption("Optional: search Wikipedia and arXiv for background context. This needs internet access and is not proof, but it helps ground the review.")
        if st.button("Check Wikipedia and arXiv"):
            with st.spinner("Checking external sources"):
                st.session_state.external_sources = verify_external_sources(report)

        external_sources = st.session_state.get("external_sources")
        if external_sources:
            st.info(external_sources["disclaimer"])
            st.write(f"Queries: {', '.join(external_sources.get('queries', []))}")

            for source_name in ("wikipedia", "arxiv"):
                results = external_sources.get(source_name, [])
                if results:
                    with st.expander(f"{source_name.title()} results", expanded=source_name == "wikipedia"):
                        for result in results[:6]:
                            if result.get("error"):
                                st.warning(f"{result['query']}: {result['error']}")
                                continue
                            st.markdown(f"**[{clean_ui_text(result['title'])}]({result.get('url', '')})**")
                            st.caption(clean_snippet_text(result.get("snippet", ""), limit=300))
                else:
                    st.caption(f"No {source_name.title()} results returned.")

        if st.checkbox("Show diagnostic details"):
            with st.expander("Skeptical findings", expanded=True):
                for finding in report["skeptical_findings"]:
                    if isinstance(finding, dict):
                        severity = finding.get("severity", "note").upper()
                        st.write(f"- **{severity}:** {clean_ui_text(finding.get('finding', finding))}")
                    else:
                        st.write(f"- {clean_ui_text(finding)}")

            with st.expander("Claims reviewed"):
                for claim in report.get("claims", [])[:8]:
                    st.write(
                        f"- **{claim.get('risk_level', 'unknown').upper()} risk:** "
                        f"{clean_ui_text(claim.get('claim', claim))}"
                    )

            with st.expander("Evidence readout"):
                for item in report.get("evidence", [])[:3]:
                    matches = item.get("matches", [])
                    if matches:
                        display_match = choose_display_match(matches)
                        score = float(display_match.get("score", 0))
                        strength = evidence_strength(score)
                        st.write(f"**Claim:** {clean_ui_text(item.get('claim', ''))}")
                        st.write(f"Evidence signal: **{strength}** (`{score:.3f}`)")
                        st.caption(clean_snippet_text(str(display_match.get("text", "")), limit=360))

            with st.expander("Citation checks"):
                citation_details = report.get("citation_details", [])
                if citation_details:
                    grouped = [item["citation"] for item in citation_details if item.get("count", 0) >= 3]
                    claim_bearing = [
                        item["citation"]
                        for item in citation_details
                        if "Claim-bearing" in item.get("check", "")
                    ]
                    st.write(f"Grouped citations: {', '.join(grouped[:8]) or 'none detected'}")
                    st.write(f"Claim-bearing citations: {', '.join(claim_bearing[:8]) or 'none detected'}")
                else:
                    for check in report.get("citation_checks", [])[:10]:
                        citation, label = citation_fallback_label(clean_ui_text(check))
                        st.write(f"- **{citation}**: {label}")

            with st.expander("Structured JSON report"):
                st.json(report)

        st.subheader("Talk With This Paper")
        st.caption(
            "Ask questions about the uploaded paper. Broad questions use the final review; "
            "specific questions use retrieved paper evidence."
        )
        web_search_enabled = st.checkbox(
            "Use web search when paper evidence is weak",
            value=workspace_web_search,
            help="When off, the chat will not search outside the paper.",
        )
        if workspace_web_search:
            web_search_enabled = True

        discussion_memory = st.session_state.get("discussion_memory") or initialize_discussion_memory(report)
        with st.expander("Discussion memory", expanded=False):
            st.write(f"Current section: {discussion_memory.get('current_section') or 'Not set yet'}")
            st.write(f"Covered: {', '.join(discussion_memory.get('covered', [])) or 'None yet'}")
            st.write(f"Remaining: {', '.join(discussion_memory.get('remaining', [])) or 'None'}")
            st.write(f"Important entities: {', '.join(discussion_memory.get('important_entities', [])) or 'None yet'}")

        suggested_questions = [
            ("Reading guide", "how should i go through it"),
            ("Paper overview", "what does the paper say"),
            ("Contributions", "what are the technical contributions"),
            ("Weaknesses", "what are the weaknesses"),
        ]
        suggestion_columns = st.columns(len(suggested_questions))
        for index, (label, suggested_question) in enumerate(suggested_questions):
            if suggestion_columns[index].button(label, key=f"suggested_question_{index}"):
                st.session_state.pending_chat_question = suggested_question

        if st.button("Clear chat"):
            st.session_state.paper_chat_messages = []
            st.session_state.discussion_memory = initialize_discussion_memory(report)
            st.session_state.pending_chat_question = ""
            st.rerun()

        for message in st.session_state.paper_chat_messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
                if message["role"] == "assistant":
                    fallback_source = {
                        "report": "Final review",
                        "heuristic": "Paper evidence search",
                        "guarded": "Paper evidence search",
                        "web": "Web plus paper evidence",
                        "web_error": "Web search error",
                        "openai": "LLM with paper evidence",
                        "ollama": "LLM with paper evidence",
                    }.get(message.get("mode"), "Paper chat")
                    sources = message.get("sources") or [fallback_source]
                    if message.get("confidence"):
                        confidence_text = f"Confidence: {message['confidence']}"
                        if message.get("confidence_reason"):
                            confidence_text += f" - {message['confidence_reason']}"
                        st.caption(confidence_text)
                    st.caption(f"Answer sources: {', '.join(sources)}")
                for evidence in message.get("evidence", [])[:2]:
                    st.caption(f"Evidence ({evidence['score']:.3f}): {clean_snippet_text(evidence['text'], limit=260)}")
                for web_result in message.get("web_results", [])[:2]:
                    if web_result.get("error"):
                        st.caption(f"Web error: {web_result.get('source', 'Web')} - {web_result['error']}")
                    else:
                        st.caption(
                            f"Web: {clean_ui_text(web_result.get('title', 'Untitled source'))} "
                            f"{web_result.get('url', '')}"
                        )
                for web_error in message.get("web_errors", [])[:2]:
                    st.caption(f"Web search status: {web_error}")

        question = st.session_state.get("pending_chat_question") or st.chat_input("Ask the paper a question")
        st.session_state.pending_chat_question = ""
        if question:
            st.session_state.paper_chat_messages.append({"role": "user", "content": question})
            response = answer_paper_question(
                question,
                st.session_state.get("paper_chat_store"),
                st.session_state.get("paper_report"),
                discussion_memory=st.session_state.get("discussion_memory"),
                web_search_enabled=web_search_enabled,
            )
            st.session_state.discussion_memory = update_discussion_memory(
                st.session_state.get("discussion_memory"),
                question,
                response,
                st.session_state.get("paper_report"),
            )
            st.session_state.paper_chat_messages.append(
                {
                    "role": "assistant",
                    "content": response["answer"],
                    "evidence": response.get("evidence", []),
                    "web_results": response.get("web_results", []),
                    "web_errors": response.get("web_errors", []),
                    "mode": response.get("mode"),
                    "confidence": response.get("confidence"),
                    "confidence_reason": response.get("confidence_reason"),
                    "sources": response.get("sources", []),
                }
            )
            st.rerun()
    else:
        st.info("Click **Review paper** to prepare the final review and enable paper chat.")
