STRUCTURAL = "structural"
ASSOCIATION = "association"
ASSERTED_FACT = "asserted_fact"

EDGE_FAMILY = {
    "MENTIONS": STRUCTURAL,
    "CO_OCCURS_WITH": ASSOCIATION,
    "ASSERTS": ASSERTED_FACT,
}


def family_of(edge_type: str) -> str:
    return EDGE_FAMILY[edge_type]
