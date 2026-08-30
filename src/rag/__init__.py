"""RAG sobre legislação brasileira (CDC).

Pipeline: ingest -> chunking -> embeddings -> store -> retrieval -> generation,
com `eval` medindo recuperação e geração de ponta a ponta.
"""
