"""Avaliação (cidadã de primeira classe deste projeto).

- Recuperação (objetiva): recall@k, MRR, nDCG@k, hit-rate contra o ground truth em evalset/.
- Geração (LLM-as-judge): groundedness, relevância e acurácia de citação via Gemini.
- RAGAS como cross-check opcional (extra `eval`).
"""
