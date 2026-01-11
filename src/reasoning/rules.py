import re

def simple_rules(claim: str):
    # Example rule: if claim contains absolute numbers/time but no units -> uncertain
    if re.search(r"\b\d+\b", claim) and not re.search(r"\b(am|pm|kg|km|%)\b", claim.lower()):
        return {"rule_flag": "MISSING_UNIT", "rule_action": "DOWNGRADE"}
    return {"rule_flag": None, "rule_action": "PASS"}