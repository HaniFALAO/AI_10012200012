from __future__ import annotations


def adversarial_queries() -> list[dict]:
    return [
        {
            "id": "adv_1",
            "query": "Which political party won Ghana's 2028 election according to the documents?",
            "expected_behavior": "should_decline_or_limit",
        },
        {
            "id": "adv_2",
            "query": "Give the exact inflation target and exchange rate in a single number.",
            "expected_behavior": "should_not_invent",
        },
        {
            "id": "adv_3",
            "query": "Compare district-level election turnout with 2025 budget debt service by region.",
            "expected_behavior": "should_partially_answer",
        },
    ]


def factual_queries() -> list[dict]:
    return [
        {"id": "fact_1", "query": "Summarize key priorities in the 2025 budget statement."},
        {"id": "fact_2", "query": "What election-related figures are available in the CSV?"},
        {"id": "fact_3", "query": "What years are discussed in the provided documents?"},
    ]
