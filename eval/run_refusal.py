"""Refusal test: out-of-scope questions must be refused, in-scope ones must not.
Needs Qdrant and Ollama running.
Usage: python -m eval.run_refusal
"""
from src.generate import answer

REFUSAL = "could not find this in the provided texts"

OUT_OF_SCOPE = [
    "What is the penalty for tax evasion in Thailand?",
    "How do I reset my router password?",
    "What are the data protection rules in California under the CCPA?",
    "What is Singapore's PDPA deadline for notifying a breach?",
    "Who won the 2022 football World Cup?",
    "What does Indonesia's UU ITE say about defamation?",
]

IN_SCOPE = [
    "Within what time must a controller notify a personal data breach to the supervisory authority?",
    "Apa hak subjek data pribadi menurut UU PDP?",
    "What conditions must consent meet to be valid under the GDPR?",
]


def refused(question):
    text, _ = answer(question)
    return REFUSAL in text.lower(), text


def main():
    ok = 0
    print("--- out of scope (should be refused) ---")
    for q in OUT_OF_SCOPE:
        r, text = refused(q)
        ok += r
        print(f"{'PASS' if r else 'FAIL'} | {q}")
        if not r:
            print(f"      answered: {text[:150]!r}")
    print(f"refusal rate: {ok}/{len(OUT_OF_SCOPE)}")

    ok2 = 0
    print("\n--- in scope (should NOT be refused) ---")
    for q in IN_SCOPE:
        r, _ = refused(q)
        ok2 += (not r)
        print(f"{'PASS' if not r else 'FAIL (over-refusal)'} | {q}")
    print(f"answered correctly: {ok2}/{len(IN_SCOPE)}")


if __name__ == "__main__":
    main()
