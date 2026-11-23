# Allergen-Aware Skincare Regimen Builder (course project, due Dec 6)
See Project_Spec_Revised.pdf for the full design.

- Data: Open Beauty Facts subset, ingested by regimen/data/ingest.py
- Safety rule: allergens and prescription conflicts are filtered in code only. The LLM
  only parses user input and rewrites reason codes; it never makes safety decisions.
- Inference: smoothed log-odds suspicion score with a known-allergen prior.
  Baselines: random, raw frequency, unsmoothed odds ratio.
- The synthetic user generator must not import anything from the scoring code.
- UI: Streamlit. Keep code simple; I need to understand and explain every part.
- Schedule: rule engine by Nov 25, scorer and generator by Nov 30, experiments by Dec 2.