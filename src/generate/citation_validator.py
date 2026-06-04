"""Parse and validate citations in LLM-generated answers.

Required citation format: [filename.pdf, page N]
Example: [rich_dad_poor_dad.pdf, page 67]
"""
import re
from dataclasses import dataclass, field

CITATION_PATTERN = re.compile(
    r"\[([^,\[\]]+?\.[a-zA-Z0-9]+),\s*page\s+(\d+)\]",
    re.IGNORECASE,
)

REFUSAL_MARKERS = (
    "i don't have enough information",
    "i do not have enough information",
    "not in the available books",
    "not in the available sources",
)


@dataclass
class Citation:
    source: str
    page: int
    raw: str


@dataclass
class ValidationResult:
    parsed: list[Citation] = field(default_factory=list)
    valid: list[Citation] = field(default_factory=list)
    invalid: list[Citation] = field(default_factory=list)
    is_refusal: bool = False
    is_valid: bool = False


def extract_citations(text: str) -> list[Citation]:
    """Extract all [source, page N] citations from text."""
    citations: list[Citation] = []
    for match in CITATION_PATTERN.finditer(text):
        citations.append(Citation(
            source=match.group(1).strip().lower(),
            page=int(match.group(2)),
            raw=match.group(0),
        ))
    return citations


def is_refusal(answer: str) -> bool:
    """Check whether the answer is a 'no information' refusal."""
    lower = answer.lower()
    return any(marker in lower for marker in REFUSAL_MARKERS)


def validate_citations(
    answer: str,
    retrieved_chunks: list[dict],
) -> ValidationResult:
    """Validate that every citation in `answer` references a retrieved chunk.

    An answer is `is_valid` if either:
      - It is a refusal ("I don't have enough information ...")
      - It contains >=1 citation AND every citation matches a retrieved chunk
    """
    parsed = extract_citations(answer)
    retrieved_set: set[tuple[str, int]] = {
        (c["source"].lower(), int(c["page_num"]))
        for c in retrieved_chunks
    }

    valid: list[Citation] = []
    invalid: list[Citation] = []
    for c in parsed:
        if (c.source, c.page) in retrieved_set:
            valid.append(c)
        else:
            invalid.append(c)

    refusal = is_refusal(answer)
    if refusal:
        is_valid_flag = True
    elif not parsed:
        is_valid_flag = False
    else:
        is_valid_flag = len(invalid) == 0

    return ValidationResult(
        parsed=parsed,
        valid=valid,
        invalid=invalid,
        is_refusal=refusal,
        is_valid=is_valid_flag,
    )
