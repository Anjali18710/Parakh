# Rules alone vs Parakh on IBM's AML data (18 of 30 done so far)

30 screening alerts from the IBM slice, half on accounts IBM labels as laundering and half on innocent ones. Model: `openai/gpt-oss-120b`. Answer key: IBM's `is_laundering` labels.

| | Laundering caught | Innocent cleared | False alarms | Precision | Right overall |
|---|---|---|---|---|---|
| Rules alone (score >= 2) | 4 of 8 | 8 of 10 | 2 | 67% | 12 of 18 |
| Parakh | 7 of 8 | 2 of 10 | 8 | 47% | 9 of 18 |

- Pattern named correctly: 0 of 7 caught laundering accounts (accounts whose IBM type is in Parakh's library).
- Claims that passed the fact-checker: 54 of 54; 10 report(s) were sent back for a rewrite.
- Average per case: 16,459 tokens, 111.2 s.
