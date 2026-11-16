# Observability and Tracing Setup for AI Hospital System using LangSmith
import os
import sys
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv

# Ensure environment variables are loaded
load_dotenv()

# Set default LangSmith environment variables if not present
os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
os.environ.setdefault("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
os.environ.setdefault("LANGCHAIN_PROJECT", "ai-hospital-sys")

try:
    from langsmith import Client, traceable
    from langsmith.run_helpers import get_current_run_tree
except ImportError:
    Client = None
    traceable = lambda *args, **kwargs: (lambda f: f)
    get_current_run_tree = lambda: None


def is_langsmith_configured() -> bool:
    """Check if LangSmith environment variables are present and enabled."""
    tracing = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() in ("true", "1")
    api_key = os.getenv("LANGCHAIN_API_KEY")
    return bool(tracing and api_key and api_key.startswith("lsv2_"))


def get_langsmith_client() -> Optional[Any]:
    """Retrieve an authenticated LangSmith client if available."""
    if not is_langsmith_configured() or Client is None:
        return None
    try:
        return Client(
            api_url=os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com"),
            api_key=os.getenv("LANGCHAIN_API_KEY"),
        )
    except Exception as e:
        print(f"[LangSmith] Client initialization warning: {e}")
        return None


def get_project_dashboard_url() -> str:
    """Generate the dashboard URL for the current LangSmith project."""
    project_name = os.getenv("LANGCHAIN_PROJECT", "ai-hospital-sys")
    return f"https://smith.langchain.com/projects?project={project_name}"


def create_workflow_config(
    run_name: str = "Hospital-AI-Governance-Workflow",
    tags: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Generate a LangChain/LangGraph execution config with LangSmith tracing metadata."""
    default_tags = ["ai-hospital-sys", "langgraph", "healthcare-rag", "llama-3.3"]
    if tags:
        default_tags.extend(tags)

    default_metadata = {
        "project": os.getenv("LANGCHAIN_PROJECT", "ai-hospital-sys"),
        "environment": "production" if os.getenv("ENV") == "prod" else "development",
        "primary_model": "llama-3.3-70b-versatile",
        "reranker_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "knowledge_base": "Hospital SOPs & Clinical Protocols",
    }
    if metadata:
        default_metadata.update(metadata)

    return {
        "run_name": run_name,
        "tags": list(set(default_tags)),
        "metadata": default_metadata,
    }


def log_user_feedback(run_id: str, score: float, comment: Optional[str] = None) -> bool:
    """Log user feedback (e.g. 1.0 for thumbs up, 0.0 for thumbs down) to LangSmith."""
    client = get_langsmith_client()
    if not client or not run_id:
        return False
    try:
        client.create_feedback(
            run_id=run_id,
            key="user-feedback",
            score=score,
            comment=comment,
        )
        return True
    except Exception as e:
        print(f"[LangSmith] Feedback logging error: {e}")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("AI Hospital System - LangSmith Observability Status")
    print("=" * 60)
    print(f"LANGCHAIN_TRACING_V2 : {os.getenv('LANGCHAIN_TRACING_V2')}")
    print(f"LANGCHAIN_PROJECT    : {os.getenv('LANGCHAIN_PROJECT')}")
    print(f"LANGCHAIN_ENDPOINT   : {os.getenv('LANGCHAIN_ENDPOINT')}")
    print(f"API Key Present      : {bool(os.getenv('LANGCHAIN_API_KEY'))}")
    print(f"LangSmith Configured : {is_langsmith_configured()}")
    print(f"Project Dashboard    : {get_project_dashboard_url()}")
    print("=" * 60)
