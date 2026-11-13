# Rules alone vs Parakh on IBM's AML data (9 of 10 done so far)

10 screening alerts from the IBM slice, half on accounts IBM labels as laundering and half on innocent ones. Model: `openai/gpt-oss-120b`. Code version: `9c937ced1561`. Answer key: IBM's `is_laundering` labels.

| | Laundering caught | Innocent cleared | False alarms | Precision | Right overall |
|---|---|---|---|---|---|
| Rules alone (score >= 2) | 2 of 5 | 2 of 4 | 2 | 50% | 4 of 9 |
| Parakh | 2 of 5 | 4 of 4 | 0 | 100% | 6 of 9 |

- Pattern named correctly: 0 of 1 caught laundering accounts (accounts whose IBM type is in Parakh's library).
- Claims that passed the fact-checker: 28 of 28; 8 report(s) were sent back for a rewrite.
- Average per case: 18,407 tokens, 132.4 s.
