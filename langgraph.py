# complete langgraph workflow using retrieval.py, hybrid_search.py, evidence_grade.py, and main.py
import json
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Optional, TypedDict

# Fix Windows console charmap encoding errors
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# -------------------------------------------------------------------------------------------------
# PREVENT RECURSION ERROR & NAMESPACE COLLISION:
# When this file is named 'langgraph.py', Python's default import mechanism attempts to import
# this local file instead of the third-party 'langgraph' package.
# We temporarily exclude the local workspace directory from sys.path to import the real package.
# -------------------------------------------------------------------------------------------------
_current_dir = os.path.dirname(os.path.abspath(__file__))
_orig_sys_path = list(sys.path)
try:
    sys.path = [p for p in sys.path if os.path.abspath(p or ".") != _current_dir]
    from langgraph.graph import StateGraph, START, END
finally:
    sys.path = _orig_sys_path
# -------------------------------------------------------------------------------------------------

from dotenv import load_dotenv
from langchain_groq import ChatGroq

# Import components from existing project modules
from guardrails import guardrail_middleware, GuardrailResult, PIIEntity
from observability import create_workflow_config, traceable, get_project_dashboard_url
from retrieval import decompose_query
from hybrid_search import get_hybrid_engine
from evidence_grade import grade_evidence

# Load environment variables
load_dotenv()

# Prevent network warning logs from LangSmith when offline
if os.getenv("LANGCHAIN_TRACING_V2") == "true":
    try:
        import socket
        socket.create_connection(("api.smith.langchain.com", 443), timeout=0.5)
    except OSError:
        os.environ["LANGCHAIN_TRACING_V2"] = "false"

# Initialize Groq LLM
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


def get_groq_llm(model: str = "llama-3.3-70b-versatile", temperature: float = 0.1) -> Optional[ChatGroq]:
    """Factory to retrieve configured ChatGroq instance."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return None
    try:
        return ChatGroq(
            model=model,
            api_key=api_key,
            temperature=temperature,
            max_tokens=2500,
            request_timeout=35.0,
        )
    except Exception as e:
        print(f"[Groq LLM Init] Warning: Unable to initialize ChatGroq: {e}")
        return None


llm: Optional[ChatGroq] = get_groq_llm()


# Define Workflow State
class HospitalWorkflowState(TypedDict):
    """Represents the complete state of the hospital AI assistant workflow."""
    query: str
    original_query: Optional[str]
    guardrail_status: Optional[dict[str, Any]]
    decomposition_plan: Optional[dict[str, Any]]
    access_status: Optional[str]
    retrieved_evidence: Optional[str]
    retrieved_docs: Optional[list[dict[str, Any]]]
    evidence_grade: Optional[dict[str, Any]]
    retry_count: int
    clarification_message: Optional[str]
    final_response: Optional[str]


# Node 0: Input Guardrails & Safety Node (PII Detection, Prompt Injection, Harmful Content)
def input_guardrail_node(state: HospitalWorkflowState) -> dict[str, Any]:
    """Screen user query for prompt injection attacks, harmful content, and PII."""
    raw_query = state.get("query", "")
    print(f"\n[Node 0: Guardrails] Screening user query for safety & PII: '{raw_query[:70]}...'")

    guard_res: GuardrailResult = guardrail_middleware.validate_input(raw_query)

    status_data = {
        "input_safe": guard_res.is_safe,
        "violation_category": guard_res.violation_category,
        "block_reason": guard_res.block_reason,
        "pii_detected": [e.model_dump() for e in guard_res.pii_detected],
        "original_query": raw_query,
        "sanitized_query": guard_res.sanitized_text,
        "output_safe": True,
        "output_pii_detected": [],
    }

    if not guard_res.is_safe:
        print(f"[Node 0: Guardrails] BLOCKED! Violation: {guard_res.violation_category} | Reason: {guard_res.block_reason}")
        return {
            "guardrail_status": status_data,
            "original_query": raw_query,
            "final_response": guard_res.advisory_response,
        }

    if guard_res.pii_detected:
        pii_types = [e.entity_type for e in guard_res.pii_detected]
        print(f"[Node 0: Guardrails] PII Detected & Redacted: {pii_types}")

    return {
        "query": guard_res.sanitized_text,
        "original_query": raw_query,
        "guardrail_status": status_data,
    }


def guardrail_block_node(state: HospitalWorkflowState) -> dict[str, Any]:
    """Terminal node for safety and guardrail violations."""
    return {
        "final_response": state.get("final_response") or "Inquiry blocked by hospital security guardrails."
    }


# Node 1: Query Decomposition Node (using retrieval.py)
def decompose_query_node(state: HospitalWorkflowState) -> dict[str, Any]:
    """Analyze and decompose the user query into atomic subtasks and access requirements."""
    query = state["query"]
    print(f"\n[Node 1: Decomposition] Analyzing user query: '{query}'")

    plan = decompose_query(query)
    intent = plan.get("primary_intent", "Unknown Intent")
    complexity = plan.get("complexity", "SIMPLE")
    subtasks = plan.get("subtasks", [])

    print(f"[Node 1: Decomposition] Intent: {intent} | Complexity: {complexity} | Subtasks: {len(subtasks)}")

    return {
        "decomposition_plan": plan,
        "retry_count": state.get("retry_count", 0),
    }


# Node 2: Clarification Node
def clarification_node(state: HospitalWorkflowState) -> dict[str, Any]:
    """Handle ambiguous or incomplete queries by requesting specific missing parameters."""
    plan = state.get("decomposition_plan", {})
    missing_items = plan.get("missing_information", ["Specific request details"])
    missing_str = ", ".join(missing_items)

    message = (
        f"Unable to process request due to missing information: {missing_str}.\n"
        "Please provide the required details (e.g. Patient ID, procedure name, or target date) to proceed."
    )
    print(f"[Node 2: Clarification] Generated clarification prompt: {message}")
    return {"clarification_message": message, "final_response": message}


# Node 3: Access Control & Authorization Node
def access_control_node(state: HospitalWorkflowState) -> dict[str, Any]:
    """Verify authorization boundary for queries involving protected health information."""
    plan = state.get("decomposition_plan", {})
    subtasks = plan.get("subtasks", [])

    access_tasks = [t for t in subtasks if t.get("execution_type") == "ACCESS_CONTROL"]
    if access_tasks:
        auth_note = (
            f"Authorization Check Flagged: Request involves protected patient records. "
            f"Access-control verification assigned to downstream security layer ({access_tasks[0].get('description')})."
        )
    else:
        auth_note = "Access Control: Standard hospital public/operational policy access authorized."

    print(f"[Node 3: Access Control] {auth_note}")
    return {"access_status": auth_note}


# Node 4: Hybrid Retrieval Node (using hybrid_search.py)
def hybrid_retrieval_node(state: HospitalWorkflowState) -> dict[str, Any]:
    """Execute dense ChromaDB + sparse BM25 + Cross-Encoder rerank retrieval."""
    query = state["query"]
    plan = state.get("decomposition_plan", {})

    # Extract specific retrieval subtask descriptions if available
    retrieval_subtasks = [
        t.get("description") for t in plan.get("subtasks", [])
        if t.get("execution_type") == "DOCUMENT_RETRIEVAL"
    ]
    search_query = f"{query} {' '.join(retrieval_subtasks)}".strip() if retrieval_subtasks else query

    print(f"[Node 4: Hybrid Search] Executing Dense + BM25 + Cross-Encoder retrieval for: '{search_query}'")
    engine = get_hybrid_engine()
    results = engine.search(search_query, top_k_candidates=10, top_n_reranked=4)

    evidence_text_lines = []
    for r in results:
        header = f"[Source: {r['source']} | Doc ID: {r['document_id']} | Dept: {r['department']} | Score: {r['score']}]"
        evidence_text_lines.append(f"{header}\n{r['content']}")

    evidence_formatted = "\n\n---\n\n".join(evidence_text_lines)
    print(f"[Node 4: Hybrid Search] Retrieved and reranked {len(results)} authoritative passages.")

    return {
        "retrieved_docs": results,
        "retrieved_evidence": evidence_formatted,
    }


# Node 5: Evidence Grade Node (using evidence_grade.py)
def grade_evidence_node(state: HospitalWorkflowState) -> dict[str, Any]:
    """Evaluate whether retrieved evidence is 'good' or 'weak' with structured reasoning."""
    query = state["query"]
    evidence = state.get("retrieved_evidence", "")

    print("[Node 5: Evidence Evaluation] Evaluating evidence relevance and policy grounding...")
    grade_result = grade_evidence(query, evidence)
    grade = grade_result.get("grade", "good")
    reasoning = grade_result.get("reasoning", "")

    print(f"[Node 5: Evidence Evaluation] Grade: '{grade.upper()}' | Reasoning: {reasoning}")
    return {"evidence_grade": grade_result}


# Node 6: Refine Search Node
def refine_search_node(state: HospitalWorkflowState) -> dict[str, Any]:
    """Refine search query with missing terms and increment retry counter."""
    retry_count = state.get("retry_count", 0) + 1
    grade_info = state.get("evidence_grade", {})
    missing_info = grade_info.get("missing_information") or "hospital clinical guidelines"

    refined_query = f"{state['query']} {missing_info}".strip()
    print(f"[Node 6: Refine Search] Retry {retry_count}: Refining search query -> '{refined_query}'")

    return {
        "query": refined_query,
        "retry_count": retry_count,
    }


def synthesize_answer_with_groq(
    query: str,
    evidence: str,
    access_status: str,
    plan: dict[str, Any],
    retrieved_docs: list[dict[str, Any]],
) -> Optional[str]:
    """Frame an authoritative, structured hospital advisory answer using Groq API and retrieved context."""
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        return None

    # Fast connectivity check to prevent hanging if running offline or in sandbox
    try:
        import socket
        socket.create_connection(("api.groq.com", 443), timeout=0.8)
    except OSError:
        return None

    # Format document reference catalog
    citations_catalog = []
    if retrieved_docs:
        for idx, doc in enumerate(retrieved_docs, start=1):
            citations_catalog.append(
                f"[{idx}] {doc.get('document_id', 'DOC')} - {doc.get('source', 'Hospital Policy')} "
                f"({doc.get('department', 'General')}) | Relevance Score: {doc.get('score', 0.0):.4f}"
            )
    citations_text = "\n".join(citations_catalog) if citations_catalog else "Approved Hospital Policy Library"

    subtasks = plan.get("subtasks", [])
    subtasks_formatted = "\n".join(
        [f"- [{t.get('task_id')}] ({t.get('execution_type')}): {t.get('description')}" for t in subtasks]
    ) if subtasks else "Single-step inquiry"

    system_prompt = (
        "You are an authoritative AI Hospital Governance, Clinical Operations, and Policy Advisory Assistant.\n"
        "Your role is to frame a comprehensive, accurate, and actionable answer to the user's healthcare inquiry, "
        "strictly grounded in the provided Hospital Policy & Protocol context.\n\n"
        "MANDATORY GOVERNANCE & ANSWER FRAMING RULES:\n"
        "1. STRICT FACTUAL GROUNDING: Rely exclusively on the provided retrieved context. Do not invent drug dosages, "
        "operational guidelines, clinical thresholds, or hospital rules not present in the evidence.\n"
        "2. INLINE CITATIONS: For every rule, protocol, workflow step, or clinical requirement, cite the governing document "
        "using its Document ID and title, e.g. `[HSP-SOP-001: Hospital Standard Operating Procedures]` or `[HSP-CLN-002: Clinical Protocols]`.\n"
        "3. INSTITUTIONAL POLICY VS. LIVE PATIENT STATUS SEPARATION:\n"
        "   - Clearly separate general hospital policies and clinical protocols from individual patient status.\n"
        "   - If the user inquiry mentions a specific patient or clinical operational action (such as surgery readiness, "
        "lab results, discharge orders, bed assignment):\n"
        "     a) State all required protocol checkpoints and documentation criteria mandated by hospital policy.\n"
        "     b) Explicitly declare that live patient status, lab completions, surgeon availability, and billing clearances "
        "CANNOT be verified from policy documents alone and require authorized lookup in the Electronic Medical Record (EMR), "
        "Laboratory Information System (LIS), or Operating Room scheduling system.\n"
        "4. STRUCTURED CLINICAL FORMAT:\n"
        "   Organize your response using clean, professional Markdown with these distinct sections:\n"
        "   - **### 🏥 Executive Summary**: Direct, definitive resolution answering the user's query.\n"
        "   - **### 📋 Governing Hospital Policies & Standards**: Detailed, bulleted explanation of all relevant rules, "
        "timelines, and protocols with inline citations.\n"
        "   - **### 🩺 Operational Clinical Checklist / Action Items**: Step-by-step required operational or clinical checks.\n"
        "   - **### 🔒 Access Control & Live Verification Status**: Necessary security clearances and pending EMR/LIS verifications.\n"
        "   - **### 📚 Verified Knowledge Base Citations**: Complete reference list of cited hospital documents.\n"
        "5. COMPLIANCE & SAFETY NOTICE: End with an advisory note emphasizing that clinical and operational decisions "
        "must be confirmed by credentialed clinical staff."
    )

    user_prompt = (
        f"### USER INQUIRY:\n{query}\n\n"
        f"### PRIMARY OBJECTIVE & INTENT:\n{plan.get('primary_intent', query)}\n"
        f"Complexity: {plan.get('complexity', 'MODERATE')}\n\n"
        f"### IDENTIFIED OPERATIONAL SUBTASKS:\n{subtasks_formatted}\n\n"
        f"### ACCESS CONTROL & SECURITY STATUS:\n{access_status}\n\n"
        f"### RETRIEVED HOSPITAL KNOWLEDGE BASE EVIDENCE:\n"
        f"```text\n{evidence}\n```\n\n"
        f"### AUTHORIZED KNOWLEDGE SOURCES:\n{citations_text}\n\n"
        f"Please synthesize the final authoritative hospital response strictly following the governance guidelines."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    # Try primary model: llama-3.3-70b-versatile; fallback to llama-3.1-8b-instant if needed
    for model_name in ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]:
        try:
            groq_llm = ChatGroq(
                model=model_name,
                api_key=groq_api_key,
                temperature=0.1,
                max_tokens=2500,
                request_timeout=35.0,
            )
            response = groq_llm.invoke(messages)
            if response and response.content and response.content.strip():
                print(f"[Generate Answer Node] Successfully synthesized answer with Groq ({model_name}).")
                return response.content.strip()
        except Exception as e:
            print(f"[Generate Answer Node] Groq ({model_name}) attempt encountered error: {e}")

    return None


def synthesize_grounded_fallback(
    query: str,
    evidence: str,
    access_status: str,
    plan: dict[str, Any],
    retrieved_docs: list[dict[str, Any]],
) -> str:
    """Generate a structured, grounded clinical governance answer when offline or when LLM is unavailable."""
    intent = plan.get("primary_intent", query)
    subtasks = plan.get("subtasks", [])

    # Format sources and evidence blocks
    doc_sections = []
    citations = []
    for d in retrieved_docs:
        doc_id = d.get("document_id", "DOC")
        source = d.get("source", "Hospital Policy")
        dept = d.get("department", "Clinical Affairs")
        score = d.get("score", 0.0)
        citations.append(f"- **{doc_id}**: *{source}* — {dept} (Rerank score: `{score:.4f}`)")

        content_clean = d.get("content", "").strip()
        doc_sections.append(
            f"#### 📄 [{doc_id}] {source} ({dept})\n"
            f"{content_clean}\n"
        )

    sources_str = "\n".join(citations) if citations else "- Approved Hospital SOP & Clinical Guidelines"
    evidence_str = "\n".join(doc_sections) if doc_sections else evidence

    subtask_bullets = "\n".join(
        [f"- **[{t.get('task_id')}]** `{t.get('execution_type')}`: {t.get('description')}" for t in subtasks]
    ) if subtasks else f"- Retrieve and synthesize hospital policy guidance for: {query}"

    return (
        f"### 🏥 Executive Summary\n\n"
        f"Regarding your inquiry: **\"{query}\"**\n\n"
        f"**Primary Objective:** {intent}\n\n"
        f"---\n\n"
        f"### 📋 Governing Hospital Policies & Standards\n\n"
        f"{evidence_str}\n\n"
        f"---\n\n"
        f"### 🩺 Operational Subtasks & Workflow Checklist\n\n"
        f"{subtask_bullets}\n\n"
        f"---\n\n"
        f"### 🔒 Access Control & Live Verification Status\n\n"
        f"**Current Status:** {access_status}\n\n"
        f"> ⚠️ **Clinical Grounding Notice:** Retrieved policy standards establish administrative and clinical "
        f"protocols. Any patient-specific determinations (such as live surgery readiness, laboratory completion, "
        f"medication orders, or discharge approval) require authenticated queries to the Electronic Medical Record (EMR) "
        f"and Laboratory Information System (LIS) with verified clinical user credentials.\n\n"
        f"---\n\n"
        f"### 📚 Verified Knowledge Base Citations\n\n"
        f"{sources_str}\n\n"
        f"> **Governance Disclaimer:** Synthetic demonstration policy rules applied. All clinical and operational "
        f"actions must comply with current hospital protocols and be confirmed by authorized personnel."
    )


# Node 7: Generate Answer Node (using retrieved evidence, metadata, and governance rules)
def generate_answer_node(state: HospitalWorkflowState) -> dict[str, Any]:
    """Synthesize final response using Groq API with strict citations, policy separation, and governance standards."""
    query = state.get("query", "")
    evidence = state.get("retrieved_evidence", "")
    access_status = state.get("access_status", "Standard policy access authorized.")
    plan = state.get("decomposition_plan", {})
    retrieved_docs = state.get("retrieved_docs", []) or []

    print("[Node 7: Answer Generation] Synthesizing authoritative hospital response with Groq API...")

    # Primary: Groq API synthesis
    answer = synthesize_answer_with_groq(
        query=query,
        evidence=evidence,
        access_status=access_status,
        plan=plan,
        retrieved_docs=retrieved_docs,
    )

    # Fallback: Grounded structured synthesis if Groq is offline or encounters error
    if not answer:
        print("[Node 7: Answer Generation] Groq offline or unavailable, generating structured grounded fallback.")
        answer = synthesize_grounded_fallback(
            query=query,
            evidence=evidence,
            access_status=access_status,
            plan=plan,
            retrieved_docs=retrieved_docs,
        )

    # Output Guardrail Screening (PII leak prevention and harmful advice filtering)
    output_guard = guardrail_middleware.validate_output(answer)
    cleaned_answer = output_guard.sanitized_text
    guard_status = dict(state.get("guardrail_status") or {})
    guard_status["output_safe"] = output_guard.is_safe
    guard_status["output_pii_detected"] = [e.model_dump() for e in output_guard.pii_detected]
    if not output_guard.is_safe:
        cleaned_answer = output_guard.advisory_response or cleaned_answer

    return {
        "final_response": cleaned_answer,
        "guardrail_status": guard_status,
    }


# Router 0: Decision after input guardrails
def router_after_guardrails(state: HospitalWorkflowState) -> str:
    """Route based on input guardrail screening."""
    guard_status = state.get("guardrail_status") or {}
    if not guard_status.get("input_safe", True):
        return "guardrail_block_node"
    return "decompose_query_node"


# Conditional Edge 1: Decision after query decomposition
def router_after_decomposition(state: HospitalWorkflowState) -> str:
    """Route based on missing information and access control requirements."""
    plan = state.get("decomposition_plan", {})
    if plan.get("missing_information"):
        return "clarification_node"
    if plan.get("access_control_required"):
        return "access_control_node"
    return "hybrid_retrieval_node"


# Conditional Edge 2: Decision after evidence grading
def router_after_grading(state: HospitalWorkflowState) -> str:
    """Route based on evidence grade and retry limits."""
    grade_info = state.get("evidence_grade", {})
    grade = grade_info.get("grade", "good")
    retry_count = state.get("retry_count", 0)

    # Refine search up to 2 times if evidence is weak
    if grade == "weak" and retry_count < 2:
        return "refine_search_node"
    return "generate_answer_node"


# Build the Complete StateGraph Workflow
workflow = StateGraph(HospitalWorkflowState)

# Add Nodes
workflow.add_node("input_guardrail_node", input_guardrail_node)
workflow.add_node("guardrail_block_node", guardrail_block_node)
workflow.add_node("decompose_query_node", decompose_query_node)
workflow.add_node("clarification_node", clarification_node)
workflow.add_node("access_control_node", access_control_node)
workflow.add_node("hybrid_retrieval_node", hybrid_retrieval_node)
workflow.add_node("grade_evidence_node", grade_evidence_node)
workflow.add_node("refine_search_node", refine_search_node)
workflow.add_node("generate_answer_node", generate_answer_node)

# Add Edges
workflow.add_edge(START, "input_guardrail_node")
workflow.add_conditional_edges(
    "input_guardrail_node",
    router_after_guardrails,
    {
        "guardrail_block_node": "guardrail_block_node",
        "decompose_query_node": "decompose_query_node",
    },
)
workflow.add_edge("guardrail_block_node", END)
workflow.add_conditional_edges(
    "decompose_query_node",
    router_after_decomposition,
    {
        "clarification_node": "clarification_node",
        "access_control_node": "access_control_node",
        "hybrid_retrieval_node": "hybrid_retrieval_node",
    },
)
workflow.add_edge("access_control_node", "hybrid_retrieval_node")
workflow.add_edge("hybrid_retrieval_node", "grade_evidence_node")
workflow.add_conditional_edges(
    "grade_evidence_node",
    router_after_grading,
    {
        "refine_search_node": "refine_search_node",
        "generate_answer_node": "generate_answer_node",
    },
)
workflow.add_edge("refine_search_node", "hybrid_retrieval_node")
workflow.add_edge("generate_answer_node", END)
workflow.add_edge("clarification_node", END)

# Compile Workflow Application
app = workflow.compile()


if __name__ == "__main__":
    print("=" * 80)
    print("AI Hospital System - Complete LangGraph Multi-Agent Workflow")
    print("=" * 80)

    test_queries = [
        "What are the pre-registration and planned admission procedures and bed allocation workflows?",
        "Check whether patient P1024 is ready for surgery tomorrow. Verify pre-operative checklist, lab reports, and surgeon availability.",
    ]

    for idx, test_query in enumerate(test_queries, start=1):
        print(f"\n{'#' * 30} TEST QUERY {idx} {'#' * 30}")
        print(f"User Query: {test_query}")

        initial_state: HospitalWorkflowState = {
            "query": test_query,
            "decomposition_plan": None,
            "access_status": None,
            "retrieved_evidence": None,
            "retrieved_docs": None,
            "evidence_grade": None,
            "retry_count": 0,
            "clarification_message": None,
            "final_response": None,
        }

        config = create_workflow_config(
            run_name=f"Hospital-MultiAgent-Query-{idx}",
            tags=["cli-test", "hospital-workflow"],
            metadata={"test_query_index": idx, "query": test_query},
        )
        output_state = app.invoke(initial_state, config=config)

        print("\n" + "=" * 30 + " FINAL WORKFLOW RESULT " + "=" * 30)
        print(output_state.get("final_response", ""))
