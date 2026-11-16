# evidence grade agent which grades sources as good or weak with reasoning using Pydantic structured output
import json
import os
import re
import sys
from typing import Any, Literal, Optional

# Fix Windows console charmap encoding errors
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()

# Prevent network warning logs from LangSmith when offline
if os.getenv("LANGCHAIN_TRACING_V2") == "true":
    try:
        import socket
        socket.create_connection(("api.smith.langchain.com", 443), timeout=0.5)
    except OSError:
        os.environ["LANGCHAIN_TRACING_V2"] = "false"

EVIDENCE_GRADE_SYSTEM_PROMPT = """You are the Senior Evidence Evaluation Agent in an AI-powered Hospital Information and Governance System.

Your core responsibility is to critically evaluate whether the retrieved hospital knowledge-base sources, clinical protocols, SOPs, and policies provide sufficient, relevant, and trustworthy evidence to answer the user's healthcare or operational query.

EVALUATION CRITERIA:
1. Relevance & Precision: Does the evidence directly address the specific question, procedure, workflow, or department asked by the user?
2. Authority & Grounding: Are the facts derived from recognized hospital documentation (e.g., Hospital SOPs, Clinical Protocols, Nursing Procedures, Admission/Discharge Guidelines)?
3. Sufficiency: Does the evidence provide actionable, complete steps or criteria, rather than vague mentions or irrelevant boilerplate notices?
4. Safety & Separation: Does the evidence clearly state policy rules without confusing general hospital policies with unverified patient-specific records?

GRADING STANDARD:
- 'good': The retrieved sources contain specific, accurate, and sufficient policy or protocol details to answer the user's query factually and safely.
- 'weak': The retrieved evidence is off-topic, generic, missing critical procedural steps, conflicting, or fails to address the user's requirements.

REASONING GUIDELINES:
- State clearly which specific sections, policies, or topics in the evidence support or fail the query.
- If grading 'weak', explicitly identify what information is missing.
- Keep the reasoning concise, objective, and analytical.
"""


class EvidenceGrade(BaseModel):
    """Structured assessment of retrieved evidence quality."""
    grade: Literal["good", "weak"] = Field(
        ...,
        description="Select 'good' if the evidence is relevant, specific, and sufficient to answer the query; select 'weak' if it is insufficient, off-topic, or ambiguous."
    )
    reasoning: str = Field(
        ...,
        description="Clear, detailed explanation justifying the grade based on content relevance and completeness."
    )
    relevant_sources: list[str] = Field(
        default_factory=list,
        description="List of document names, IDs, or sections that directly support the answer."
    )
    missing_information: Optional[str] = Field(
        default=None,
        description="Specific information or policies required by the query that were not found in the evidence."
    )


class EvidenceGradeAgent:
    """Agent that grades retrieved evidence as good or weak with structured reasoning."""

    def __init__(
        self,
        model: str = "llama-3.3-70b-versatile",
        api_key: Optional[str] = None,
    ) -> None:
        self.api_key: Optional[str] = api_key or os.getenv("GROQ_API_KEY")
        self.model: str = model
        self.llm: Optional[ChatGroq] = None
        if self.api_key:
            try:
                self.llm = ChatGroq(
                    model=self.model,
                    api_key=self.api_key,
                    temperature=0.0,
                )
            except Exception:
                self.llm = None

    def _fallback_heuristic_grade(self, query: str, evidence: str) -> EvidenceGrade:
        """Heuristic rule-based grading when offline or when LLM is unavailable."""
        cleaned_query = query.lower()
        cleaned_evidence = evidence.lower()

        # Stop words filter for query keywords
        stop_words = {
            "what", "is", "the", "are", "and", "or", "for", "to", "in", "of",
            "a", "an", "how", "does", "do", "can", "with", "from", "by", "patient"
        }
        query_words = [
            w for w in re.findall(r"\b[a-zA-Z]{3,}\b", cleaned_query)
            if w not in stop_words
        ]

        if not query_words or not evidence.strip():
            return EvidenceGrade(
                grade="weak",
                reasoning="Empty query or evidence provided. Cannot verify evidence quality.",
                relevant_sources=[],
                missing_information="Sufficient query or evidence text.",
            )

        # Count keyword matches
        matches = [w for w in query_words if w in cleaned_evidence]
        match_ratio = len(matches) / len(query_words) if query_words else 0.0

        # Extract document IDs from evidence if present
        found_sources = list(
            dict.fromkeys(
                re.findall(r"\bHSP-[A-Z0-9-]+\b", evidence)
                + re.findall(r"\b(?:0\d_[A-Za-z_]+\.pdf)\b", evidence)
            )
        )

        has_substantive_length = len(evidence.strip()) > 150

        if match_ratio >= 0.5 and has_substantive_length:
            return EvidenceGrade(
                grade="good",
                reasoning=(
                    f"Retrieved evidence covers {len(matches)}/{len(query_words)} key query concepts "
                    f"({', '.join(matches[:5])}) and references authorized hospital documentation."
                ),
                relevant_sources=found_sources,
                missing_information=None,
            )
        else:
            missing = [w for w in query_words if w not in matches]
            return EvidenceGrade(
                grade="weak",
                reasoning=(
                    f"Evidence has low conceptual coverage ({len(matches)}/{len(query_words)} terms matched). "
                    f"Key missing topics: {', '.join(missing[:4])}."
                ),
                relevant_sources=found_sources,
                missing_information=f"Policies covering {', '.join(missing[:4])}",
            )

    def grade(self, query: str, evidence: str) -> dict[str, Any]:
        """Grade the evidence as good or weak and return structured output."""
        if self.llm:
            try:
                structured_judge = self.llm.with_structured_output(EvidenceGrade)
                messages = [
                    {"role": "system", "content": EVIDENCE_GRADE_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            f"User Query:\n{query}\n\n"
                            f"Retrieved Evidence:\n{evidence}"
                        ),
                    },
                ]
                result: EvidenceGrade = structured_judge.invoke(messages)
                return result.model_dump()
            except Exception:
                # Graceful fallback to heuristic judge if API is unreachable
                pass

        fallback_result = self._fallback_heuristic_grade(query, evidence)
        return fallback_result.model_dump()


def grade_evidence(query: str, evidence: str) -> dict[str, Any]:
    """Convenience function to evaluate and grade evidence quality."""
    agent = EvidenceGradeAgent()
    return agent.grade(query, evidence)


@tool
def grade_evidence_tool(query: str, evidence: str) -> str:
    """
    Grade retrieved hospital evidence as 'good' or 'weak' with detailed reasoning.
    Returns structured JSON output containing grade, reasoning, relevant_sources, and missing_information.
    """
    result = grade_evidence(query, evidence)
    return json.dumps(result, indent=2)


# Aliases for tool naming flexibility
evidence_grade_tool = grade_evidence_tool
evidence_grader = grade_evidence_tool


if __name__ == "__main__":
    sample_query = "What are the pre-registration and planned admission procedures?"
    sample_good_evidence = (
        "[Source: 04_Admission_Policies.pdf | Doc ID: HSP-ADM-004]\n"
        "02. Pre-registration and planned admission\n"
        "Planned admissions must be scheduled in advance through the hospital coordination system. "
        "Identity verification requires government-issued photo ID and insurance policy details. "
        "A pre-admission checklist must be completed 24 hours prior to patient arrival."
    )
    sample_weak_evidence = (
        "[Source: 01_Hospital_Standard_Operating_Procedures.pdf | Doc ID: HSP-SOP-001]\n"
        "08. Facilities, equipment and downtime\n"
        "Air handling units and HVAC systems must undergo routine quarterly maintenance by engineering staff."
    )

    print("=" * 70)
    print("Testing Evidence Grade Tool (@tool)")
    print("=" * 70)
    print(f"Query: {sample_query}\n")

    print("--- Test 1: Tool Invocation (Relevant Evidence) ---")
    tool_output_1 = grade_evidence_tool.invoke({
        "query": sample_query,
        "evidence": sample_good_evidence,
    })
    print(tool_output_1)

    print("\n--- Test 2: Tool Invocation (Weak/Irrelevant Evidence) ---")
    tool_output_2 = grade_evidence_tool.invoke({
        "query": sample_query,
        "evidence": sample_weak_evidence,
    })
    print(tool_output_2)
