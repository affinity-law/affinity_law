"""Exact byte revisions of the two released rule tables with translated display labels.

Row order, rule IDs, thresholds and all statistical fields match in each pair.
Other revisions and all non-table inputs retain their original integrity checks.
"""

ENGLISH_TABLES = {
    "10ea195660ae76ab3cec3710e0cb063f081a9431ea3986415f42cb14022a9ee9": "6bd3cc3337983fce7db0e0457efc82686b830bacd8f8c03cc5d25208ceb217cf",
    "c748a50f521c3380c357d523325fe4fe7f34270d1377841e5a5e972ff97d07e4": "2084e57e8a1c6279b2a5d2d02e05351292c912e7d9bcfaa74b3b8a192f831a7a",
}


def equivalent_selection(signature):
    return {key: ENGLISH_TABLES.get(value, value) if key in ("rules", "pooled") else value
            for key, value in signature.items()}
