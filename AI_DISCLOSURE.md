# AI Disclosure

## AI/ML in the submitted system

The retrieval system uses the frozen
`sentence-transformers/all-MiniLM-L6-v2` model to encode natural-language
queries and code snippets into semantic embeddings. It runs inference on CPU
and ranks snippets using embedding similarity. The model is used for retrieval;
it is not a generative model in this pipeline.

## Development assistance

OpenAI Codex was used as a coding assistant during implementation, evaluation
tooling, documentation, and submission packaging under developer direction.
This disclosure does not mean that AI generated the entire project. Codex is
not required at runtime and is not part of the submitted retrieval pipeline.

## Generative AI

No LLM or generative-AI service is used by the submitted inference pipeline.
It returns ranked code snippets and does not generate code or natural-language
answers.
