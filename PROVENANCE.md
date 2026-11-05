# CLIO offline demo provenance

This package contains a **derived demonstration adapter**, not the CLIO engine. Its scenarios are synthetic. The review page replays four outputs captured by running the existing evaluator; it does not call the evaluator, a model, a broker or a market feed.

## Source and evidence

- CLIO source commit: `46da00609029298c2bca6d7d0f723b103f50272f`.
- Evaluator source SHA256: `21baef96a8b9c62521435a3c4b54753fb82323cd5c84a4f1e5937078b7fbd017`.
- Direction source file SHA256: `6dac5127b10cc03614836b1e03c91fd49088bbd958a846cc75a6995183b22a47`.
- Price confirmation source file SHA256: `b70e88590c4e501edb644e0c3d96b367f9a9debfcc06d9d5bbbc64580f770344`.
- Captured source manifest SHA256: `a898ac06d1daac492b38143b4755a03be92199caaa75462f1fe1ab0dcc61fc79`.
- Existing-source synthetic run output SHA256: `3661341d7f2a30952c0ff4c741dcd2f78d32f9531028c63c384385e6391d9655`.
- Selected historical model-record source SHA256: `1061ef7cb997af7c53a1b6a87f5d764336940a90fc78f9210a16251ff1ca6450`. `META_EVIDENCE.json` contains a reduced categorical projection without private source paths.

## Validation

Root ran the unmodified source evaluator with four synthetic fixtures; the observed dispositions were `WOULD_DEFER`, `RECONFIRMED`, `INSUFFICIENT` and `NO_REENTRY_CONTEXT`. No provider, model, broker or order call was made. The execution used a process-local audit hook and a network-denial negative control; this is not a claim of operating-system isolation.

`demo_core.py` reproduces selected unmodified syntax-tree bodies from the evaluator, the existing price predicate, and the Direction enum. `run_demo.py` supplies synthetic objects and uses the included IANA time-zone data. Root independently checked all 11 selected syntax-tree nodes and ran the derived adapter on Python 3.14: all four complete case objects matched the unmodified source run. The saved parity record SHA256 is `a15761f9d1ebe7bbc0f19050eabfdcacd390287da6521d8dfa169889927c358a`; adapter stdout SHA256 is `166f1198a51fd14f0c96c2443152aecc6fcb269e5e73b87e93a96860f1d8204d`. The network-denial negative control passed in that process; this is not an OS isolation claim. The local page shows the saved original-source outputs, not outputs freshly calculated in the browser.

IANA time-zone data is accompanied by `tzdata/TZDATA_COPYRIGHT.txt`. CLIO was built by Dominic Newman with AI assistance. Original CLIO demonstration code is licensed under MIT; see LICENSE and LICENSE_SCOPE.md. Brand/media/evidence materials and third-party notices retain their separate scope. AI assistance and source provenance are disclosed here; competition eligibility is not certified by this license. No financial performance, live signal, new model inference or avoided loss is claimed.
