from __future__ import annotations

"""Score dbt schema descriptions, resolving doc() references from markdown docs."""

# Load environment variables FIRST, before other imports
import os
from pathlib import Path

# Load .env file before pydantic_ai import
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, skip loading .env

# Now import the rest
import argparse
import asyncio
import json
import logging
import re
import sys
import time
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    import yaml
except ImportError as exc:  # pragma: no cover - clear runtime message
    raise SystemExit(
        "Missing dependency 'pyyaml'. Install with: pip install pyyaml"
    ) from exc


DEFAULT_PLACEHOLDERS = {
    "tbd",      # To Be Determined
    "todo",     # Unfinished task
    "n/a",      # Not applicable
    "na",       # Not applicable (variant)
    "none",     # Placeholder
    "lorem",    # Lorem ipsum text
    "ipsum",    # Lorem ipsum text
    "placeholder",  # Obvious placeholder
    "unknown",  # Vague placeholder
    "fixme",    # Unfinished marker
}

VAGUE_WORDS = {
    "data",         # Too generic
    "value",        # Too generic
    "info",         # Too generic
    "information",  # Too generic
    "stuff",        # Vague
    "thing",        # Vague
    "various",      # Lacks specificity
    "etc",          # Incomplete
    "misc",         # Miscellaneous is vague
    "miscellaneous",# Vague
}

REDUNDANT_PHRASES = [
    "this column",
    "this field",
    "this is",
    "contains the",
    "stores the",
]


class Column(BaseModel):
    """dbt column definition within a model.
    
    Attributes:
        name: Column name (required)
        description: Column description or doc() reference (optional)
        tests: List of dbt tests applied to this column (optional)
    """
    name: str
    description: Optional[str] = None
    tests: Optional[List[Any]] = None


class Model(BaseModel):
    """dbt model definition with optional column metadata.
    
    Attributes:
        name: Model name (required)
        description: Model description or doc() reference (optional)
        columns: List of Column definitions (optional)
    """
    name: str
    description: Optional[str] = None
    columns: List[Column] = Field(default_factory=list)


class DbtSchema(BaseModel):
    """Root dbt schema.yml structure (models only).
    
    Attributes:
        models: List of Model definitions
    """
    models: List[Model] = Field(default_factory=list)


class Issue(BaseModel):
    """A single quality issue found in a description.
    
    Attributes:
        severity: CRITICAL, HIGH, MEDIUM, or LOW
        message: Human-readable issue description
        deduction: Points deducted from score
    """
    severity: str
    message: str
    deduction: int


class DescriptionAssessment(BaseModel):
    """Assessment result for a single description string.
    
    Attributes:
        score: Quality score from 0-100
        issues: List of Issue objects found
        suggestions: List of specific improvement suggestions
    """
    score: int
    issues: List[Issue]
    suggestions: List[str] = Field(default_factory=list)


def _contains_placeholder(text: str, placeholders: Iterable[str]) -> bool:
    """Return True if any placeholder token appears in the description.
    
    Performs case-insensitive substring matching.
    
    Args:
        text: Text to search
        placeholders: Iterable of placeholder strings to look for
        
    Returns:
        True if any placeholder is found as lowercase substring
        
    Example:
        _contains_placeholder("Status is TBD", ["tbd"]) -> True
    """
    lowered = text.lower()
    return any(ph in lowered for ph in placeholders)


# Regex pattern: matches {{ doc('name') }} or {{ doc("name") }}
# Used to find dbt doc() references in description strings
DOC_REF_RE = re.compile(r"\{\{\s*doc\(['\"](?P<name>[^'\"]+)['\"]\)\s*\}\}")

# Regex pattern: matches {% docs name %} ... {% enddocs %} blocks
# Used to extract documentation from markdown files
DOC_BLOCK_RE = re.compile(
    r"\{%\s*docs\s+(?P<name>\w+)\s*%\}(?P<body>.*?)\{%\s*enddocs\s*%\}",
    re.DOTALL,  # Allow . to match newlines
)


def parse_docs_from_markdown(text: str) -> Dict[str, str]:
    """Extract dbt docs blocks into a {name: body} mapping.
    
    Parses dbt-style documentation blocks from markdown:
    
    {% docs my_doc %}
    This is the documentation text.
    {% enddocs %}
    
    Args:
        text: Raw markdown content
        
    Returns:
        Dictionary mapping doc names to their trimmed body text
        
    Example:
        docs = parse_docs_from_markdown(md_content)
        # docs = {"my_doc": "This is the documentation text."}
    """
    docs: Dict[str, str] = {}
    for match in DOC_BLOCK_RE.finditer(text):
        name = match.group("name")
        body = match.group("body").strip()
        docs[name] = body
    return docs


def load_docs(paths: Sequence[Path]) -> Dict[str, str]:
    """Load docs blocks from one or more markdown files.
    
    Args:
        paths: Sequence of Path objects to markdown files
        
    Returns:
        Merged dictionary of all {name: body} from all files.
        Later files override earlier files for duplicate names.
        
    Example:
        docs = load_docs([Path("schema_docs.md"), Path("extra_docs.md")])
    """
    docs: Dict[str, str] = {}
    for path in paths:
        if not path.exists():
            continue
        docs.update(parse_docs_from_markdown(path.read_text(encoding="utf-8")))
    return docs


def resolve_description(
    text: Optional[str], docs: Dict[str, str]
) -> Tuple[Optional[str], List[Issue]]:
    """
    Resolve {{ doc('name') }} references to docs content, return issues.
    
    Args:
        text: Description that may contain doc() references
        docs: Dictionary of {name: body} from loaded markdown docs
        
    Returns:
        (resolved_text, issues) where issues is empty if doc found,
        or contains Issue if doc reference is missing.
    """
    if text is None:
        return None, []

    match = DOC_REF_RE.search(text)
    if not match:
        # No doc() reference found; return text as-is
        return text, []

    doc_name = match.group("name")
    doc_text = docs.get(doc_name)
    if doc_text is None:
        # Doc reference found but docs block not loaded
        return None, [Issue(
            severity="HIGH",
            message=f"missing docs block: {doc_name}",
            deduction=0
        )]
    # Successfully resolved doc reference
    return doc_text, []


# Initialize Pydantic AI assessment agent with Ollama (local, no API key needed)
try:
    # Agent with generic type parameter for result type
    # The generic type Agent[DescriptionAssessment, None] tells Pydantic AI what to return
    assessment_agent: Agent[DescriptionAssessment, None] = Agent(
        'ollama:qwen2.5',  # Use Ollama Qwen 2.5 model (local, faster and higher quality than llama3.2)
        system_prompt="""You are an expert documentation quality assessor for database schemas.

To provide structured output, follow this JSON format in your response:
{
  "score": <0-100 integer>,
  "issues": [
    {"severity": "CRITICAL|HIGH|MEDIUM|LOW", "message": "<description>", "deduction": <0-100 int>}
  ],
  "suggestions": ["<suggestion1>", "<suggestion2>", "<suggestion3>"]
}

Evaluate descriptions based on these criteria:
- CRITICAL: Missing description (score 0), placeholder text like TODO/TBD/N/A
- HIGH: Too short (<20 chars), too generic (missing specificity)
- MEDIUM: Too long (>300 chars), vague/generic words, redundant phrases, formatting issues
- LOW: Missing capitalization, missing ending punctuation

Be concise and practical in suggestions."""
    )
    AI_ENABLED = True
    print(f"✅ Pydantic AI initialized successfully with Ollama Qwen 2.5", file=sys.stderr)
except Exception as e:
    assessment_agent = None
    AI_ENABLED = False
    print(f"❌ Failed to initialize Pydantic AI: {e}", file=sys.stderr)


async def assess_description_ai(
    text: Optional[str],
    *,
    min_len: int = 20,
    max_len: int = 300,
    placeholders: Sequence[str] = tuple(DEFAULT_PLACEHOLDERS),
    vague_words: Sequence[str] = tuple(VAGUE_WORDS),
    redundant_phrases: Sequence[str] = REDUNDANT_PHRASES,
) -> Optional[DescriptionAssessment]:
    """
    Score a description using Pydantic AI-powered assessment.
    
    Uses Ollama (local model) via Pydantic AI to intelligently evaluate descriptions.
    Returns None if text is empty or AI agent is unavailable.
    
    Args:
        text: Description to assess
        min_len: Minimum length in characters (default 20)
        max_len: Maximum length in characters (default 300)
        placeholders: Tuple of strings to flag as placeholders
        vague_words: Tuple of vague words to detect
        redundant_phrases: List of redundant phrases to detect
        
    Returns:
        DescriptionAssessment with AI-powered score, issues, and suggestions
        or None if prerequisites are not met
    """
    if text is None or not text.strip() or not AI_ENABLED or assessment_agent is None:
        return None
    
    # Use Pydantic AI for assessment
    prompt = f"""You are a database documentation quality expert. Assess this dbt description:

DESCRIPTION: "{text}"

CONSTRAINTS:
- Minimum length: {min_len} characters
- Maximum length: {max_len} characters
- Avoid vague words: {', '.join(list(vague_words)[:8])}
- Avoid redundant phrases: {', '.join(redundant_phrases)}
- Placeholder text is critical issue: {', '.join(list(placeholders)[:8])}

RATING SCALE:
- CRITICAL (0-40): Missing, placeholder, severely vague
- HIGH (41-60): Too short, missing context, generic
- GOOD (61-100): Specific, well-formatted, helpful

Provide score 0-100, list issues with severity (CRITICAL/HIGH/MEDIUM/LOW) and deductions, and 2-3 improvement suggestions."""

    result = await assessment_agent.run(prompt)
    # Parse the string output into DescriptionAssessment object
    import json
    try:
        # result.data is the structured output from the LLM
        if hasattr(result, 'data') and isinstance(result.data, DescriptionAssessment):
            return result.data
        # If result.output is a string, try to parse it as JSON
        elif isinstance(result.output, str):
            try:
                json_data = json.loads(result.output)
                return DescriptionAssessment(**json_data)
            except (json.JSONDecodeError, TypeError):
                # If JSON parsing fails, return None so fallback is used
                return None
        else:
            # If it's already the right type, return it
            return result.output if isinstance(result.output, DescriptionAssessment) else None
    except Exception as e:
        print(f"⚠️  Error parsing AI response: {e}", file=sys.stderr)
        return None


def assess_description_with_ai(
    text: Optional[str],
    *,
    min_len: int = 20,
    max_len: int = 300,
    placeholders: Sequence[str] = tuple(DEFAULT_PLACEHOLDERS),
    vague_words: Sequence[str] = tuple(VAGUE_WORDS),
    redundant_phrases: Sequence[str] = REDUNDANT_PHRASES,
) -> DescriptionAssessment:
    """
    Assess description using AI-powered evaluation (Pydantic AI only).
    
    Uses Ollama (local model) via Pydantic AI for intelligent assessment.
    Requires Ollama to be running locally (ollama serve).
    
    This function ONLY uses AI assessment - no rule-based fallback.
    If AI is unavailable, returns a low score with an error message.
    
    Args:
        text: Description to assess
        min_len: Minimum length in characters (default 20)
        max_len: Maximum length in characters (default 300)
        placeholders: Tuple of strings to flag as placeholders
        vague_words: Tuple of vague words to detect
        redundant_phrases: List of redundant phrases to detect
        
    Returns:
        DescriptionAssessment with AI-powered score, issues, and suggestions
        or error assessment if AI is unavailable
    """
    # Check if AI is available
    if not AI_ENABLED or assessment_agent is None:
        return DescriptionAssessment(
            score=0,
            issues=[Issue(
                severity="CRITICAL",
                message="AI assessment unavailable - ensure Ollama is running (ollama serve)",
                deduction=100
            )],
            suggestions=["Start Ollama server with 'ollama serve' and ensure qwen2.5 model is installed"]
        )
    
    # Handle empty/missing text
    if text is None or not text.strip():
        return DescriptionAssessment(
            score=0,
            issues=[Issue(severity="CRITICAL", message="missing description", deduction=100)],
            suggestions=["Add a description that explains the purpose and content"]
        )
    
    # Use AI assessment only
    try:
        # Run async function from sync context
        result = asyncio.run(
            assess_description_ai(
                text,
                min_len=min_len,
                max_len=max_len,
                placeholders=placeholders,
                vague_words=vague_words,
                redundant_phrases=redundant_phrases,
            )
        )
        if result is not None:
            return result
        else:
            # AI returned None (shouldn't happen but handle it)
            return DescriptionAssessment(
                score=0,
                issues=[Issue(
                    severity="CRITICAL",
                    message="AI assessment failed to return result",
                    deduction=100
                )],
                suggestions=["Check AI model configuration and API key"]
            )
    except Exception as e:
        # AI call failed - return error instead of falling back
        return DescriptionAssessment(
            score=0,
            issues=[Issue(
                severity="CRITICAL",
                message=f"AI assessment error: {str(e)}",
                deduction=100
            )],
            suggestions=["Check that Ollama is running (ollama serve) and qwen2.5 model is installed (ollama pull qwen2.5)"]
        )


def assess_description(
    text: Optional[str],
    *,
    min_len: int = 20,
    max_len: int = 300,
    placeholders: Sequence[str] = tuple(DEFAULT_PLACEHOLDERS),
    vague_words: Sequence[str] = tuple(VAGUE_WORDS),
    redundant_phrases: Sequence[str] = REDUNDANT_PHRASES,
) -> DescriptionAssessment:
    """
    Optimized rule-based description assessment.
    
    Efficient quality evaluation using pattern matching and heuristics.
    Uses multiple deduction rules for comprehensive assessment:
    
    CRITICAL (blockers):
    - Missing description: score=0
    - Contains placeholder text: -40 points
    
    HIGH (major issues):
    - Too short (< min_len): -40 points
    - Too generic (single word or <3 words with reasonable length): -30 points
    
    MEDIUM (quality issues):
    - Too long (> max_len): -15 points
    - Contains vague words: -15 points
    - Has redundant phrases: -10 points
    - Multiple spaces or formatting issues: -10 points
    
    LOW (polish issues):
    - Missing capital letter: -5 points
    - Missing ending punctuation: -5 points
    
    Args:
        text: Description to assess
        min_len: Minimum length in characters (default 20)
        max_len: Maximum length in characters (default 300)
        placeholders: Tuple of strings to flag as placeholders
        vague_words: Tuple of vague words to detect
        redundant_phrases: List of redundant phrases to detect
        
    Returns:
        DescriptionAssessment with score, categorized issues, and suggestions
    """
    # Fast path: Missing or empty description scores 0
    if text is None or not text.strip():
        return DescriptionAssessment(
            score=0,
            issues=[Issue(severity="CRITICAL", message="missing description", deduction=100)],
            suggestions=["Add a description that explains the purpose and content"]
        )

    stripped = text.strip()
    score = 100
    issues: List[Issue] = []
    suggestions: List[str] = []
    
    # CRITICAL: Check for placeholder text (fast check)
    if _contains_placeholder(stripped, placeholders):
        deduction = 40
        score -= deduction
        issues.append(Issue(
            severity="CRITICAL",
            message="contains placeholder text (e.g., TODO, TBD, N/A)",
            deduction=deduction
        ))
        suggestions.append("Replace placeholder text with actual description")

    # HIGH: Check length constraints
    if len(stripped) < min_len:
        deduction = 40
        score -= deduction
        issues.append(Issue(
            severity="HIGH",
            message=f"too short ({len(stripped)} chars, minimum {min_len})",
            deduction=deduction
        ))
        suggestions.append(f"Expand description to at least {min_len} characters with specific details")

    # HIGH: Check for overly generic descriptions (word-based heuristic)
    word_count = len(stripped.split())
    if word_count < 3 and len(stripped) >= min_len:
        deduction = 30
        score -= deduction
        issues.append(Issue(
            severity="HIGH",
            message="overly generic ({} words)".format(word_count),
            deduction=deduction
        ))
        suggestions.append("Add more context: explain what, why, or how this is used")

    # MEDIUM: Check maximum length
    if len(stripped) > max_len:
        deduction = 15
        score -= deduction
        issues.append(Issue(
            severity="MEDIUM",
            message=f"too long ({len(stripped)} chars, maximum {max_len})",
            deduction=deduction
        ))
        suggestions.append(f"Simplify description to under {max_len} characters")

    # MEDIUM: Check for vague words (optimized substring matching)
    lowered = stripped.lower()
    found_vague = [word for word in vague_words if f" {word} " in f" {lowered} " or lowered.startswith(f"{word} ") or lowered.endswith(f" {word}")]
    if found_vague:
        deduction = 15
        score -= deduction
        issues.append(Issue(
            severity="MEDIUM",
            message=f"contains vague words: {', '.join(found_vague[:3])}",
            deduction=deduction
        ))
        suggestions.append("Replace vague words with specific terminology")

    # MEDIUM: Check for redundant phrases (efficient substring matching)
    found_redundant = [phrase for phrase in redundant_phrases if phrase in lowered]
    if found_redundant:
        deduction = 10
        score -= deduction
        issues.append(Issue(
            severity="MEDIUM",
            message=f"contains redundant phrases: {', '.join(found_redundant[:2])}",
            deduction=deduction
        ))
        suggestions.append("Remove redundant phrases like 'this column' or 'contains the'")

    # MEDIUM: Check for formatting issues (multiple spaces, double punctuation)
    if "  " in stripped or ",," in stripped:
        deduction = 10
        score -= deduction
        issues.append(Issue(
            severity="MEDIUM",
            message="formatting issues (multiple spaces or punctuation)",
            deduction=deduction
        ))
        suggestions.append("Fix spacing and punctuation")

    # LOW: Check formatting - capital letter at start
    if not stripped[0].isupper():
        deduction = 5
        score -= deduction
        issues.append(Issue(
            severity="LOW",
            message="should start with a capital letter",
            deduction=deduction
        ))
        suggestions.append("Capitalize the first letter")

    # LOW: Check formatting - punctuation at end
    if stripped[-1] not in ".!?":
        deduction = 5
        score -= deduction
        issues.append(Issue(
            severity="LOW",
            message="should end with punctuation (.!?)",
            deduction=deduction
        ))
        suggestions.append("Add ending punctuation")

    return DescriptionAssessment(
        score=max(score, 0),
        issues=issues,
        suggestions=suggestions if issues else []
    )


def get_rating(score: int) -> str:
    """Convert numeric score to letter grade rating.
    
    Args:
        score: Quality score 0-100
        
    Returns:
        Letter grade: A (90-100), B (70-89), C (50-69), D (0-49)
    """
    if score >= 90:
        return "A"
    elif score >= 70:
        return "B"
    elif score >= 50:
        return "C"
    else:
        return "D"


class ItemResult(BaseModel):
    """Result for a single model or column assessment.
    
    Attributes:
        type: 'model' or 'column'
        name: Name of the model or column
        parent: Parent model name (for columns only)
        score: Quality score 0-100
        rating: Letter grade (A, B, C, D) based on score
        issues: List of Issue objects
        suggestions: List of improvement suggestions
    """
    type: str  # 'model' or 'column'
    name: str
    parent: Optional[str] = None  # For columns, the parent model name
    score: int
    rating: str = Field(default="")
    issues: List[Issue]
    suggestions: List[str]


class AssessmentMetrics(BaseModel):
    """Metrics for tracking assessment performance and results."""
    total_items: int
    total_models: int
    total_columns: int
    average_score: float
    excellent_count: int  # 90-100
    good_count: int  # 70-89
    fair_count: int  # 50-69
    poor_count: int  # 0-49
    critical_issues_count: int
    high_issues_count: int
    medium_issues_count: int
    low_issues_count: int
    execution_time_seconds: float
    timestamp: str


class SchemaAssessmentReport(BaseModel):
    """Complete schema assessment report with results and metrics."""
    status: str  # "success" or "error"
    metrics: AssessmentMetrics
    results: List[ItemResult]
    error_message: Optional[str] = None


def load_schema(path: Path) -> DbtSchema:
    """Load and validate a dbt schema.yml file.
    
    Args:
        path: Path to schema.yml file
        
    Returns:
        Validated DbtSchema object parsed from YAML
        
    Raises:
        pydantic.ValidationError: If YAML structure doesn't match schema
        FileNotFoundError: If file doesn't exist (caller should check)
    """
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return DbtSchema.model_validate(data)


def compute_metrics(results: List[ItemResult], execution_time: float) -> AssessmentMetrics:
    """Compute comprehensive metrics from assessment results.
    
    Args:
        results: List of ItemResult objects
        execution_time: Total execution time in seconds
        
    Returns:
        AssessmentMetrics with statistics and counts
    """
    total_items = len(results)
    total_models = len([r for r in results if r.type == "model"])
    total_columns = len([r for r in results if r.type == "column"])
    avg_score = sum(r.score for r in results) / total_items if total_items > 0 else 0
    
    excellent_count = len([r for r in results if r.score >= 90])
    good_count = len([r for r in results if 70 <= r.score < 90])
    fair_count = len([r for r in results if 50 <= r.score < 70])
    poor_count = len([r for r in results if r.score < 50])
    
    # Count issues by severity
    critical_issues = sum(len([i for i in r.issues if i.severity == "CRITICAL"]) for r in results)
    high_issues = sum(len([i for i in r.issues if i.severity == "HIGH"]) for r in results)
    medium_issues = sum(len([i for i in r.issues if i.severity == "MEDIUM"]) for r in results)
    low_issues = sum(len([i for i in r.issues if i.severity == "LOW"]) for r in results)
    
    return AssessmentMetrics(
        total_items=total_items,
        total_models=total_models,
        total_columns=total_columns,
        average_score=round(avg_score, 2),
        excellent_count=excellent_count,
        good_count=good_count,
        fair_count=fair_count,
        poor_count=poor_count,
        critical_issues_count=critical_issues,
        high_issues_count=high_issues,
        medium_issues_count=medium_issues,
        low_issues_count=low_issues,
        execution_time_seconds=round(execution_time, 3),
        timestamp=datetime.utcnow().isoformat() + "Z"
    )


def assess_schema(schema: DbtSchema, docs: Dict[str, str]) -> SchemaAssessmentReport:
    """
    Assess schema quality and return structured report.
    
    Uses Pydantic AI assessment to evaluate all descriptions and returns
    structured metrics and results suitable for both JSON output and logging.
    
    Args:
        schema: DbtSchema object parsed from YAML
        docs: Dictionary of {name: body} from loaded markdown docs
        
    Returns:
        SchemaAssessmentReport with metrics, results, and status
    """
    start_time = time.time()
    
    # Check AI availability upfront
    if not AI_ENABLED or assessment_agent is None:
        logger.error("AI assessment unavailable - Ollama not running or qwen2.5 not installed")
        return SchemaAssessmentReport(
            status="error",
            metrics=AssessmentMetrics(
                total_items=0,
                total_models=0,
                total_columns=0,
                average_score=0,
                excellent_count=0,
                good_count=0,
                fair_count=0,
                poor_count=0,
                critical_issues_count=0,
                high_issues_count=0,
                medium_issues_count=0,
                low_issues_count=0,
                execution_time_seconds=0,
                timestamp=datetime.utcnow().isoformat() + "Z"
            ),
            results=[],
            error_message="AI assessment unavailable - ensure Ollama is running (ollama serve) and qwen2.5 is installed"
        )
    
    logger.info(f"Starting schema assessment with Ollama Qwen 2.5")
    
    results: List[ItemResult] = []
    
    # Collect all assessment results
    for model in schema.models:
        logger.debug(f"Assessing model: {model.name}")
        
        # Assess model description using AI
        model_desc, model_doc_issues = resolve_description(model.description, docs)
        model_assessment = assess_description_with_ai(model_desc)
        
        # Combine doc resolution issues with assessment issues
        all_model_issues = model_doc_issues + model_assessment.issues
        
        model_result = ItemResult(
            type="model",
            name=model.name,
            score=model_assessment.score,
            rating=get_rating(model_assessment.score),
            issues=all_model_issues,
            suggestions=model_assessment.suggestions
        )
        results.append(model_result)
        
        # Assess each column with AI-powered assessment
        for col in model.columns:
            logger.debug(f"Assessing column: {model.name}.{col.name}")
            
            col_desc, col_doc_issues = resolve_description(col.description, docs)
            col_assessment = assess_description_with_ai(col_desc)
            all_col_issues = col_doc_issues + col_assessment.issues
            
            col_result = ItemResult(
                type="column",
                name=col.name,
                parent=model.name,
                score=col_assessment.score,
                rating=get_rating(col_assessment.score),
                issues=all_col_issues,
                suggestions=col_assessment.suggestions
            )
            results.append(col_result)
    
    execution_time = time.time() - start_time
    metrics = compute_metrics(results, execution_time)
    
    logger.info(f"Assessment complete: {metrics.total_items} items in {execution_time:.2f}s, avg score: {metrics.average_score}")
    
    return SchemaAssessmentReport(
        status="success",
        metrics=metrics,
        results=results
    )


def print_human_readable_report(report: SchemaAssessmentReport) -> None:
    """
    Print human-readable console report from assessment results.
    
    Args:
        report: SchemaAssessmentReport with metrics and results
    """
    if report.status == "error":
        print("❌ ERROR: Schema assessment failed!")
        print(f"   {report.error_message}")
        return
    
    metrics = report.metrics
    results = report.results
    
    print("🤖 Using Pydantic AI (Ollama Qwen 2.5) for assessment...\n")
    
    # Print Executive Summary
    print("=" * 80)
    print("🎯 SCHEMA QUALITY REPORT")
    print("=" * 80)
    print()
    
    score_ranges = {
        "Excellent (90-100)": metrics.excellent_count,
        "Good (70-89)": metrics.good_count,
        "Fair (50-69)": metrics.fair_count,
        "Poor (0-49)": metrics.poor_count,
    }
    
    print(f"📊 Summary Statistics:")
    print(f"  Total items: {metrics.total_items} (models: {metrics.total_models}, columns: {metrics.total_columns})")
    print(f"  Average score: {metrics.average_score}/100")
    print(f"  Execution time: {metrics.execution_time_seconds}s")
    print(f"  Issues found: CRITICAL={metrics.critical_issues_count}, HIGH={metrics.high_issues_count}, MEDIUM={metrics.medium_issues_count}, LOW={metrics.low_issues_count}")
    print(f"  Score distribution:")
    for range_name, count in score_ranges.items():
        pct = (count / metrics.total_items * 100) if metrics.total_items > 0 else 0
        bar = "█" * int(pct / 5)  # 20 chars max
        print(f"    {range_name:20} {count:3} ({pct:5.1f}%) {bar}")
    print()
    
    # Group items by severity for targeted attention
    critical_results = [r for r in results if any(i.severity == "CRITICAL" for i in r.issues)]
    high_results = [r for r in results if any(i.severity == "HIGH" for i in r.issues) and r not in critical_results]
    medium_results = [r for r in results if any(i.severity == "MEDIUM" for i in r.issues) and r not in critical_results and r not in high_results]
    
    # Print items needing attention
    if critical_results or high_results or medium_results:
        print("🚨 Items Requiring Attention:")
        print()
        
        if critical_results:
            print(f"  ❌ CRITICAL ({len(critical_results)} items):")
            for item in critical_results[:5]:
                location = f"{item.parent}.{item.name}" if item.parent else item.name
                critical_issues = [i.message for i in item.issues if i.severity == "CRITICAL"]
                print(f"    • {location:35} Score:{item.score:3} → {critical_issues[0]}")
            if len(critical_results) > 5:
                print(f"    ... and {len(critical_results) - 5} more")
            print()
        
        if high_results:
            print(f"  ⚠️  HIGH ({len(high_results)} items):")
            for item in high_results[:5]:
                location = f"{item.parent}.{item.name}" if item.parent else item.name
                high_issues = [i.message for i in item.issues if i.severity == "HIGH"]
                print(f"    • {location:35} Score:{item.score:3} → {high_issues[0]}")
            if len(high_results) > 5:
                print(f"    ... and {len(high_results) - 5} more")
            print()
        
        if medium_results:
            print(f"  ⚡ MEDIUM ({len(medium_results)} items):")
            for item in medium_results[:3]:
                location = f"{item.parent}.{item.name}" if item.parent else item.name
                medium_issues = [i.message for i in item.issues if i.severity == "MEDIUM"]
                print(f"    • {location:35} Score:{item.score:3} → {medium_issues[0]}")
            if len(medium_results) > 3:
                print(f"    ... and {len(medium_results) - 3} more")
            print()
    else:
        print("✅ No critical issues found!\n")


def report_schema(schema: DbtSchema, docs: Dict[str, str]) -> None:
    """
    Print comprehensive quality report using Pydantic AI assessment.
    
    Legacy function - calls assess_schema and prints human-readable output.
    For structured output, use assess_schema() directly.
    """
    report = assess_schema(schema, docs)
    print_human_readable_report(report)


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    """Parse CLI arguments for schema and docs paths.
    
    Args:
        argv: Command-line arguments (sys.argv format)
        
    Returns:
        Namespace with attributes:
        - schema (Path): Path to dbt schema.yml file
        - docs (List[Path]): Paths to markdown docs files (repeatable)
        - json_output (bool): Whether to output JSON instead of human-readable
        - output_file (Path): Optional path to write JSON output
        
    Example:
        python testing_pydantic.py sample_schema.yml
        python testing_pydantic.py sample_schema.yml --json
        python testing_pydantic.py sample_schema.yml --json --output report.json
    """
    parser = argparse.ArgumentParser(
        description="Score dbt schema descriptions with AI-powered assessment."
    )
    parser.add_argument("schema", type=Path, help="Path to dbt schema.yml")
    parser.add_argument(
        "--docs",
        type=Path,
        action="append",
        default=[],
        help="Path to dbt docs .md file (repeatable).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output structured JSON instead of human-readable format (cloud-ready)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Write JSON output to file (requires --json)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    return parser.parse_args(list(argv)[1:])


def main(argv: Sequence[str]) -> int:
    """CLI entrypoint; loads docs and schema then reports scores.
    
    Workflow:
    1. Parse command-line arguments (schema path, optional doc paths)
    2. Auto-discover .md files in schema directory if --docs not provided
    3. Load and parse all docs blocks from markdown files
    4. Load and validate schema.yml with Pydantic
    5. Assess and report scores for each model and column description
    6. Output as JSON (--json) or human-readable format (default)
    
    Returns:
        Exit code (0=success, 1=assessment issues, 2=error)
        
    Example:
        python testing_pydantic.py sample_schema.yml
        python testing_pydantic.py sample_schema.yml --json --output report.json
        python testing_pydantic.py sample_schema.yml --docs schema_docs.md --verbose
    """
    args = parse_args(argv)
    
    # Configure logging based on verbosity
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)

    path = args.schema
    if not path.exists():
        logger.error(f"Schema file not found: {path}")
        print(f"❌ ERROR: File not found: {path}", file=sys.stderr)
        return 2

    # Load docs
    docs_paths = args.docs
    if not docs_paths:
        docs_paths = sorted(path.parent.glob("*.md"))
    
    logger.info(f"Loading docs from {len(docs_paths)} file(s)")
    docs = load_docs(docs_paths)
    
    # Load and assess schema
    try:
        logger.info(f"Loading schema from {path}")
        schema = load_schema(path)
    except Exception as e:
        logger.error(f"Failed to load schema: {e}")
        print(f"❌ ERROR: Failed to load schema: {e}", file=sys.stderr)
        return 2
    
    # Run assessment
    try:
        report = assess_schema(schema, docs)
    except Exception as e:
        logger.error(f"Assessment failed: {e}")
        print(f"❌ ERROR: Assessment failed: {e}", file=sys.stderr)
        return 2
    
    # Output results
    if args.json:
        # JSON output mode (cloud-ready)
        json_output = report.model_dump_json(indent=2)
        
        if args.output:
            # Write to file
            logger.info(f"Writing JSON output to {args.output}")
            args.output.write_text(json_output, encoding="utf-8")
            print(f"✅ Report written to {args.output}", file=sys.stderr)
        else:
            # Print to stdout
            print(json_output)
    else:
        # Human-readable output mode
        print_human_readable_report(report)
    
    # Return exit code based on status and quality
    if report.status == "error":
        return 2
    elif report.metrics.critical_issues_count > 0 or report.metrics.average_score < 50:
        logger.warning(f"Quality issues detected: {report.metrics.critical_issues_count} critical issues")
        return 1
    else:
        logger.info("Assessment completed successfully")
        return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
