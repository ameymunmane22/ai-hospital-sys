# take input query and build a agent to perform query decomposition 
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

SYSTEM_PROMPT = """1. ROLE AND OBJECTIVE

You are the Query Decomposition Agent in a secure, hospital-focused AI assistant.

Your primary responsibility is to analyze the user's query and decompose complex, multi-step requests into smaller, logically connected, independently answerable subtasks.

These subtasks will be executed by downstream components, including:

Knowledge Retrieval Agent (RAG)
Patient Records / EMR Tool
Appointment Management Tool
Laboratory Information System (LIS)
Radiology Information System (RIS)
Insurance Verification Tool
Pharmacy / Medication Information Tool
Billing System
Authorization and Access-Control Layer

You do NOT answer the user's original question.
You do NOT execute tools.
You do NOT invent patient information, hospital policies, or operational facts.

You produce a structured decomposition plan for downstream execution.

2. CORE RESPONSIBILITIES

For every incoming query:
- Understand the user's primary objective.
- Identify all distinct information requirements.
- Determine whether the query requires document retrieval, deterministic tools, or both.
- Break complex requests into atomic subtasks.
- Identify dependencies between subtasks.
- Identify which subtasks can run in parallel.
- Identify missing information that prevents execution.
- Preserve the original intent and constraints of the user.
- Identify sensitive or unauthorized information requests for access-control review.
- Return a structured execution plan.

3. DECOMPOSITION RULES

Rule A: Atomic Subtasks
Each subtask must address one specific information requirement or operational action.

Rule B: Identify the Correct Execution Source
Assign exactly one primary execution type to each subtask:
- DOCUMENT_RETRIEVAL: Information available in approved hospital documents, policies, SOPs, protocols, or guidelines.
- PATIENT_RECORD_LOOKUP: Patient-specific information available through authorized records.
- DETERMINISTIC_TOOL: A structured calculation, verification, status check, or operational action.
- CLARIFICATION: Additional information is required before execution.
- ACCESS_CONTROL: Authorization must be checked before accessing protected information or performing an action.

Do not use document retrieval to establish live patient-specific status.

Rule C: Dependencies
Identify dependencies explicitly.
Use:
- depends_on: IDs of subtasks that must finish first.
- parallel_group: Tasks that can execute simultaneously.
- execution_order: Logical execution sequence.
Do not create unnecessary dependencies between independent retrieval tasks.

Rule D: Query Completeness
Determine whether the query contains sufficient information for execution.
If required information is missing, create a CLARIFICATION subtask and specify exactly what is missing.

Rule E: Multi-Source Retrieval
When a question requires information from multiple documents, create separate retrieval subtasks for each relevant source category.

Rule F: Safety and Authorization
Hospital data may contain sensitive patient information.
Flag patient-specific requests for authorization checks and requests involving protected health information.

Rule G: Avoid Over-Decomposition
Do not split simple questions unnecessarily.
Only decompose when it improves execution, accuracy, or traceability.

5. OUTPUT FORMAT
Return ONLY valid JSON matching this schema:
{
  "original_query": "User's original query",
  "primary_intent": "Short description of the user's objective",
  "complexity": "SIMPLE | MODERATE | COMPLEX",
  "requires_decomposition": true,
  "missing_information": [],
  "access_control_required": true,
  "subtasks": [
    {
      "task_id": "T1",
      "description": "Specific atomic task",
      "execution_type": "DOCUMENT_RETRIEVAL | PATIENT_RECORD_LOOKUP | DETERMINISTIC_TOOL | CLARIFICATION | ACCESS_CONTROL",
      "source": "Relevant document category, system, or tool",
      "information_required": [],
      "depends_on": [],
      "parallel_group": 1,
      "execution_order": 1,
      "priority": "HIGH | MEDIUM | LOW",
      "requires_authorization": false,
      "expected_output": "What this task should return"
    }
  ],
  "execution_plan": {
    "strategy": "PARALLEL | SEQUENTIAL | HYBRID",
    "parallel_task_groups": [["T1"]],
    "sequential_dependencies": [
      {
        "task_id": "T1",
        "depends_on": []
      }
    ]
  },
  "final_response_requirements": {
    "must_include_citations": true,
    "must_verify_policy_version": true,
    "must_distinguish_policy_from_patient_status": true,
    "must_not_claim_unverified_actions": true
  }
}

6. STRICT CONSTRAINTS
- Never answer the original user query.
- Never fabricate documents, policies, patient records, or tool capabilities.
- Never assume a document contains information that has not yet been retrieved.
- Never treat retrieved policy information as proof of patient-specific eligibility or status.
- Return valid JSON matching the specified schema.
"""


class Subtask(BaseModel):
    task_id: str = Field(description="Unique task identifier, e.g. T1, T2")
    description: str = Field(description="Specific atomic task description")
    execution_type: Literal[
        "DOCUMENT_RETRIEVAL",
        "PATIENT_RECORD_LOOKUP",
        "DETERMINISTIC_TOOL",
        "CLARIFICATION",
        "ACCESS_CONTROL",
    ] = Field(description="Execution category for downstream runner")
    source: str = Field(description="Relevant document category, system, or tool")
    information_required: list[str] = Field(
        default_factory=list, description="Specific inputs required"
    )
    depends_on: list[str] = Field(
        default_factory=list, description="List of prerequisite task IDs"
    )
    parallel_group: int = Field(
        default=1, description="Parallel execution group number"
    )
    execution_order: int = Field(default=1, description="Execution step order")
    priority: Literal["HIGH", "MEDIUM", "LOW"] = Field(
        default="HIGH", description="Priority level"
    )
    requires_authorization: bool = Field(
        default=False, description="Whether access control is required"
    )
    expected_output: str = Field(description="Expected output or artifact")


class SequentialDependency(BaseModel):
    task_id: str = Field(description="Dependent task ID")
    depends_on: list[str] = Field(
        default_factory=list, description="Prerequisite task IDs"
    )


class ExecutionPlan(BaseModel):
    strategy: Literal["PARALLEL", "SEQUENTIAL", "HYBRID"] = Field(
        description="Overall execution strategy"
    )
    parallel_task_groups: list[list[str]] = Field(
        default_factory=list, description="Groups of tasks runnable in parallel"
    )
    sequential_dependencies: list[SequentialDependency] = Field(
        default_factory=list, description="Sequential dependency mappings"
    )


class FinalResponseRequirements(BaseModel):
    must_include_citations: bool = Field(
        default=True, description="Require citations"
    )
    must_verify_policy_version: bool = Field(
        default=True, description="Verify policy versions"
    )
    must_distinguish_policy_from_patient_status: bool = Field(
        default=True, description="Distinguish policy from patient status"
    )
    must_not_claim_unverified_actions: bool = Field(
        default=True, description="Prevent unverified claims"
    )


class QueryDecompositionPlan(BaseModel):
    original_query: str = Field(description="Original user query")
    primary_intent: str = Field(description="Short description of user intent")
    complexity: Literal["SIMPLE", "MODERATE", "COMPLEX"] = Field(
        description="Complexity rating"
    )
    requires_decomposition: bool = Field(
        description="Whether query required splitting"
    )
    missing_information: list[str] = Field(
        default_factory=list, description="Information missing from query"
    )
    access_control_required: bool = Field(
        description="Whether authorization is required"
    )
    subtasks: list[Subtask] = Field(
        default_factory=list, description="Decomposed subtasks"
    )
    execution_plan: ExecutionPlan = Field(description="Execution plan")
    final_response_requirements: FinalResponseRequirements = Field(
        default_factory=FinalResponseRequirements,
        description="Verification constraints",
    )


class QueryDecompositionAgent:
    """Agent responsible for decomposing complex healthcare queries into structured execution plans."""

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

    def _fallback_heuristic_decompose(self, query: str) -> QueryDecompositionPlan:
        """Fallback decomposition when network or API is unreachable."""
        cleaned_query = query.strip()
        patient_id_match = re.search(r"\b(P\d+|patient\s+[A-Za-z0-9]+)\b", cleaned_query, re.IGNORECASE)
        patient_indicators = ["patient p", "my father", "my mother", "my son", "my daughter", "mr. ", "ms. ", "mrs. "]
        is_patient_specific = bool(patient_id_match) or any(ind in cleaned_query.lower() for ind in patient_indicators)

        if not is_patient_specific:
            # General policy/procedure question - does not need patient ID
            task = Subtask(
                task_id="T1",
                description=f"Retrieve hospital documentation addressing: {cleaned_query}",
                execution_type="DOCUMENT_RETRIEVAL",
                source="Approved hospital policies, SOPs, and clinical protocols",
                information_required=["Relevant hospital policy topics"],
                depends_on=[],
                parallel_group=1,
                execution_order=1,
                priority="HIGH",
                requires_authorization=False,
                expected_output="Applicable hospital guidelines and policy citations",
            )
            return QueryDecompositionPlan(
                original_query=cleaned_query,
                primary_intent=f"Retrieve policy documentation regarding: {cleaned_query}",
                complexity="SIMPLE",
                requires_decomposition=False,
                missing_information=[],
                access_control_required=False,
                subtasks=[task],
                execution_plan=ExecutionPlan(
                    strategy="SEQUENTIAL",
                    parallel_task_groups=[["T1"]],
                    sequential_dependencies=[],
                ),
                final_response_requirements=FinalResponseRequirements(),
            )

        # Complex query with specific patient data
        subtasks: list[Subtask] = []
        parallel_groups: list[list[str]] = []
        dependencies: list[SequentialDependency] = []
        p_id = patient_id_match.group(1) if patient_id_match else "Patient"

        # T1: Access Control
        subtasks.append(
            Subtask(
                task_id="T1",
                description=f"Verify requesting user authorization to access records for {p_id}",
                execution_type="ACCESS_CONTROL",
                source="Hospital authorization and access-control layer",
                information_required=["User identity", "Patient identifier", "Requested scope"],
                depends_on=[],
                parallel_group=1,
                execution_order=1,
                priority="HIGH",
                requires_authorization=True,
                expected_output="Authorization decision and permitted access scope",
            )
        )
        parallel_groups.append(["T1"])

        # T2: Document Retrieval for standards/protocols
        subtasks.append(
            Subtask(
                task_id="T2",
                description=f"Retrieve relevant clinical protocol and checklists for: {cleaned_query}",
                execution_type="DOCUMENT_RETRIEVAL",
                source="Approved clinical protocols and hospital SOPs",
                information_required=["Protocol category", "Checklist requirements"],
                depends_on=["T1"],
                parallel_group=2,
                execution_order=2,
                priority="HIGH",
                requires_authorization=False,
                expected_output="Applicable protocol guidelines and readiness criteria",
            )
        )

        # T3: Patient Record / Diagnostic status
        subtasks.append(
            Subtask(
                task_id="T3",
                description=f"Retrieve clinical, laboratory, and order status for {p_id}",
                execution_type="PATIENT_RECORD_LOOKUP",
                source="Electronic Medical Records (EMR) & Laboratory Information System (LIS)",
                information_required=["Patient identifier", "Required diagnostics list"],
                depends_on=["T1", "T2"],
                parallel_group=3,
                execution_order=3,
                priority="HIGH",
                requires_authorization=True,
                expected_output="Diagnostic completion status and clinical orders",
            )
        )

        # T4: Comparison / Verification
        subtasks.append(
            Subtask(
                task_id="T4",
                description="Synthesize patient findings against hospital protocol requirements",
                execution_type="DETERMINISTIC_TOOL",
                source="Readiness evaluation workflow",
                information_required=["Protocol criteria", "Patient clinical status"],
                depends_on=["T2", "T3"],
                parallel_group=4,
                execution_order=4,
                priority="HIGH",
                requires_authorization=False,
                expected_output="Evaluation report highlighting satisfied and pending criteria",
            )
        )

        parallel_groups.extend([["T2"], ["T3"], ["T4"]])
        dependencies.extend([
            SequentialDependency(task_id="T2", depends_on=["T1"]),
            SequentialDependency(task_id="T3", depends_on=["T1", "T2"]),
            SequentialDependency(task_id="T4", depends_on=["T2", "T3"]),
        ])

        return QueryDecompositionPlan(
            original_query=cleaned_query,
            primary_intent=f"Evaluate multi-factor hospital query regarding {p_id}",
            complexity="COMPLEX",
            requires_decomposition=True,
            missing_information=[],
            access_control_required=True,
            subtasks=subtasks,
            execution_plan=ExecutionPlan(
                strategy="HYBRID",
                parallel_task_groups=parallel_groups,
                sequential_dependencies=dependencies,
            ),
            final_response_requirements=FinalResponseRequirements(),
        )

    def decompose(self, query: str) -> dict[str, Any]:
        """Decompose an input query into an actionable execution plan."""
        if not query or not query.strip():
            empty_plan = QueryDecompositionPlan(
                original_query="",
                primary_intent="Empty query received",
                complexity="SIMPLE",
                requires_decomposition=False,
                missing_information=["User query text"],
                access_control_required=False,
                subtasks=[
                    Subtask(
                        task_id="T1",
                        description="Request clarification on empty input query",
                        execution_type="CLARIFICATION",
                        source="User interface",
                        information_required=["Query text"],
                        depends_on=[],
                        parallel_group=1,
                        execution_order=1,
                        priority="LOW",
                        requires_authorization=False,
                        expected_output="User query input",
                    )
                ],
                execution_plan=ExecutionPlan(
                    strategy="SEQUENTIAL",
                    parallel_task_groups=[["T1"]],
                    sequential_dependencies=[],
                ),
            )
            return empty_plan.model_dump()

        if self.llm:
            try:
                structured_llm = self.llm.with_structured_output(QueryDecompositionPlan)
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"USER QUERY:\n\n{query}"},
                ]
                result: QueryDecompositionPlan = structured_llm.invoke(messages)
                return result.model_dump()
            except Exception:
                # Graceful fallback to heuristic decomposer if network/API is unavailable
                pass

        plan = self._fallback_heuristic_decompose(query)
        return plan.model_dump()


def decompose_query(query: str) -> dict[str, Any]:
    """Top-level convenience function to decompose a query using the QueryDecompositionAgent."""
    agent = QueryDecompositionAgent()
    return agent.decompose(query)


@tool
def query_decomposition_tool(query: str) -> str:
    """Decompose complex hospital and clinical queries into a structured multi-step execution plan in JSON format."""
    plan = decompose_query(query)
    return json.dumps(plan, indent=2)


query_decomposition = query_decomposition_tool


if __name__ == "__main__":
    sample_query = (
        "Check whether patient P1024 is ready for surgery tomorrow. "
        "Verify whether all pre-operative tests are completed, insurance authorization is approved, "
        "and the surgeon is available."
    )
    print("=" * 60)
    print("Hospital Query Decomposition Tool")
    print("=" * 60)
    print(f"Input Query:\n{sample_query}\n")

    tool_result = query_decomposition_tool.invoke({"query": sample_query})
    print("Tool Output (JSON):")
    print(tool_result)
