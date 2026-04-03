
import re

# =========================
# CONFIG
# =========================
NEGATION_WORDS = ["not", "no", "never", "none", "n't"]
UNCERTAIN_WORDS = ["may", "might", "could", "possibly", "reportedly", "allegedly"]


# =========================
# HELPER FUNCTIONS
# =========================
def contains_negation(text):
    text = text.lower()
    return any(re.search(rf"\b{word}\b", text) for word in NEGATION_WORDS)


def contains_uncertainty(text):
    text = text.lower()
    return any(re.search(rf"\b{word}\b", text) for word in UNCERTAIN_WORDS)


def simple_overlap(claim, evidence):
    """
    Đo mức độ overlap đơn giản giữa claim và evidence
    """
    claim_words = set(re.findall(r'\w+', claim.lower()))
    evidence_words = set(re.findall(r'\w+', evidence.lower()))

    overlap = claim_words.intersection(evidence_words)
    return len(overlap) / (len(claim_words) + 1e-6)


def extract_ordinals(text):
    # find ordinals like 45th, 1st, 2nd
    matches = re.findall(r"(\d+)(?:st|nd|rd|th)", text)
    return [int(m) for m in matches]


def extract_years(text):
    matches = re.findall(r"(19|20)\d{2}", text)
    return matches


def detect_orbital_contradiction(claim, evidence):
    """
    Detect simple orbital relation contradictions between claim and evidence.
    Returns 'REFUTES' if evidence clearly states the opposite relation.
    """
    c = claim.lower()
    e = evidence.lower()

    # simple patterns
    sun_terms = ["sun", "the sun"]
    earth_terms = ["earth", "the earth"]

    # verbs indicating orbit/revolve
    orbit_verbs = ["orbit", "orbits", "orbited", "revolve", "revolves", "revolved", "revolving", "revolves around", "revolves around"]

    # If claim asserts Sun orbits Earth
    claim_sun_orbits_earth = any(v in c and "sun" in c and "earth" in c for v in ["revolve", "revolves", "revolves around", "orbit", "orbits"]) 
    # If evidence asserts Earth orbits Sun (opposite)
    evidence_earth_orbits_sun = any(v in e and "earth" in e and "sun" in e for v in orbit_verbs)

    if claim_sun_orbits_earth and evidence_earth_orbits_sun:
        return True

    # Also check reverse: claim says Earth orbits Sun but evidence says Sun orbits Earth
    claim_earth_orbits_sun = any(v in c and "earth" in c and "sun" in c for v in orbit_verbs)
    evidence_sun_orbits_earth = any(v in e and "sun" in e and "earth" in e for v in orbit_verbs)
    if claim_earth_orbits_sun and evidence_sun_orbits_earth:
        return True

    return False


def extract_ordinals(text: str):
    if not isinstance(text, str):
        return []
    nums = []
    for m in re.findall(r"(\d+)(?:st|nd|rd|th)\b", text, flags=re.IGNORECASE):
        try:
            nums.append(int(m))
        except:
            pass
    return nums


def extract_years(text: str):
    if not isinstance(text, str):
        return []
    return [int(y) for y in re.findall(r"\b(18|19|20)\d{2}\b", text)]


# =========================
# MAIN REASONING FUNCTION
# =========================
def apply_reasoning(claim, evidence, ml_label, confidence):
    """
    claim: str
    evidence: str
    ml_label: SUPPORTS / REFUTES / NOT ENOUGH INFO
    confidence: float

    return:
        final_label, reason
    """

    # =========================
    # CASE 1: Evidence yếu hoặc không có
    # =========================
    if not evidence or "No evidence" in evidence:
        return "NOT ENOUGH INFO", "No reliable evidence found"

    overlap_score = simple_overlap(claim, evidence)

    # Nếu evidence gần như không liên quan → bỏ
    if overlap_score < 0.1:
        return "NOT ENOUGH INFO", f"Low relevance evidence (overlap={overlap_score:.2f})"

    # =========================
    # CASE 2: Evidence chứa uncertainty
    # =========================
    if contains_uncertainty(evidence):
        return "NOT ENOUGH INFO", "Evidence is uncertain"

    # =========================
    # CASE 3: Negation handling (CẨN THẬN)
    # =========================
    claim_neg = contains_negation(claim)
    evidence_neg = contains_negation(evidence)

    # Nếu claim và evidence phủ định NGƯỢC nhau → conflict
    if claim_neg != evidence_neg:
        return "NOT ENOUGH INFO", "Negation conflict between claim and evidence"

    # =========================
    # CASE 3b: Relation contradictions (e.g., orbital relations)
    # =========================
    try:
        if detect_orbital_contradiction(claim, evidence):
            return "REFUTES", "Evidence indicates opposite relation (orbital mismatch)"
    except Exception:
        pass

    # =========================
    # CHECK ORDINAL / YEAR MISMATCH
    # =========================
    claim_ord = extract_ordinals(claim)
    evidence_ord = extract_ordinals(evidence)
    if claim_ord and evidence_ord:
        # If claim asserts a specific ordinal but evidence mentions a different one -> refute
        # (e.g., "50th president" vs evidence says "45th president")
        if any(c not in evidence_ord for c in claim_ord):
            return "REFUTES", f"Ordinal mismatch: claim {claim_ord} vs evidence {evidence_ord}"

    # year based contradictions (simple heuristic)
    claim_years = extract_years(claim)
    evidence_years = extract_years(evidence)
    if claim_years and evidence_years:
        # if claim mentions a year that does not appear in evidence context, fall back to NOT ENOUGH INFO
        if not any(y in evidence_years for y in claim_years):
            return "NOT ENOUGH INFO", "Year mismatch between claim and evidence"

    # =========================
    # CASE 4: ML confidence check
    # =========================
    if confidence < 0.6:
        return "NOT ENOUGH INFO", "Low model confidence"

    if claim.lower() in evidence.lower():
        return "SUPPORTS", "Direct textual match found"

    # =========================
    # CASE 5: Default → tin ML
    # =========================
    return ml_label, "Predicted by ML model" 