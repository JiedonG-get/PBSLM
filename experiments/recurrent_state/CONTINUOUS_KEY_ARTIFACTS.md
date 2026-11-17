# Continuous-key experiment artifact history

All files below are preserved because they were produced during staged code correction; they must not be pooled as independent seeds.

- `continuous_key_smoke.json`: 2-update pipeline smoke for seed0, not a model-quality run.
- `continuous_keys_correction800.json`: first 800-update run using the corrected `observe/query` APIs before abstention-triggered fallback metrics were added. Reports state/reset/dictionary only.
- `continuous_keys_hybrid800.json`: repeated 800-update run with fallback metrics. For all three seeds, model weight hashes and evaluation dictionaries match the later reconciled `continuous_keys_final800.json` exactly; different elapsed times/provenance metadata do not constitute independent observations.
- `continuous_keys_final800.json`: canonical current artifact used by `CONTINUOUS_KEYS.md` and its summary.
- `all_tests_continuous_keys.log`: earlier full-suite run (215 tests); `all_tests_continuous_keys_final.log` is the later final run also reporting 215 tests.

The initial broken-API invocation raised before writing an artifact; it has no performance result and is documented narratively in the experiment report. No stale artifact is used as an independent model seed.
