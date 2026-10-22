# Provenance

`protocol.json` records evaluation context and aggregation rules; `config.json` preserves resolved evaluation and checkpoint settings. `verification.json` records source and artifact SHA256 hashes. Original result/history inputs are preserved in `inputs/`.

Evaluation sidecars are checked against their CSV hashes. Legacy inputs without metadata remain explicitly unavailable; no configuration is inferred. Paired scenarios are verified using both seed and tape hash. Source hashes identify evidence and do not certify simulator validity.
