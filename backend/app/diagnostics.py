import re


def diagnose(question: str, retrieved_chunks: list[dict]) -> dict:
    if not retrieved_chunks:
        return {
            "status": "no_context",
            "issues": ["No documents have been indexed or no chunks were retrieved."],
            "recommendations": ["Upload a supported document and try the query again."],
        }

    best_score = retrieved_chunks[0]["score"]
    issues = []
    recommendations = []

    if best_score < 0.05:
        status = "retrieval_failure"
        issues.append("No retrieved chunk has meaningful lexical similarity to the question.")
        recommendations.extend([
            "Check whether the uploaded documents contain the information being asked for.",
            "Try clearer wording or domain-specific keywords.",
        ])
    elif best_score < 0.18:
        status = "weak_retrieval"
        issues.append("The best retrieved chunk has relatively low lexical similarity.")
        recommendations.extend([
            "Review the source document and chunk boundaries.",
            "Try a more specific question.",
        ])
    else:
        status = "retrieval_candidate"
        recommendations.append("Review the retrieved context to confirm it supports the answer.")

    question_terms = set(re.findall(r"\b[a-zA-Z]{3,}\b", question.lower()))
    context_terms = set()
    for item in retrieved_chunks:
        context_terms.update(re.findall(r"\b[a-zA-Z]{3,}\b", item["content"].lower()))
    coverage = len(question_terms & context_terms) / max(len(question_terms), 1)

    return {
        "status": status,
        "best_similarity_score": best_score,
        "question_term_coverage": round(coverage, 4),
        "issues": issues,
        "recommendations": recommendations,
        "note": "This MVP uses TF-IDF lexical similarity. It does not prove factual correctness or detect hallucinations.",
    }


def evaluate_retrieval(question: str, expected_answer: str, retrieved_chunks: list[dict]) -> dict:
    expected_terms = set(re.findall(r"\b[a-zA-Z]{3,}\b", expected_answer.lower()))
    context = " ".join(item["content"] for item in retrieved_chunks).lower()
    context_terms = set(re.findall(r"\b[a-zA-Z]{3,}\b", context))
    overlap = len(expected_terms & context_terms) / max(len(expected_terms), 1)

    return {
        "question": question,
        "expected_answer_term_coverage": round(overlap, 4),
        "retrieved_chunk_count": len(retrieved_chunks),
        "interpretation": (
            "Higher lexical overlap means more expected-answer terms appeared in retrieved context. "
            "This is a simple baseline metric, not a semantic or factuality score."
        ),
    }
