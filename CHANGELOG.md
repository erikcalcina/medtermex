# CHANGELOG

## medtermex-0.2.0 (2025-10-16)

### Initial release

**Supported Extraction Approaches:**

- GLiNER: A lightweight model for named entity recognition with configurable thresholds and multiple matching modes
- Unsloth: Efficient fine-tuning of LLM models with configurable temperature and model parameters
- Ollama: Integration with Ollama-based LLM models for medical term extraction

**Data Processing Pipelines:**

- Regex-based label validation for GLiNER predictions
- Regex-based label validation for LLM predictions
- LLM-based label validation pipeline for structured entity extraction
- Prompt formatting with system prompts, entity deduplication, and JSON-based output parsing
- Support for training and evaluation data formatting with batch processing

**Evaluation Metrics:**

- Exact match evaluation: Precise entity and label matching
- Relaxed match evaluation: Substring-based entity matching
- Overlap-based evaluation: Partial overlap between predicted and true entities
- BERTScore-inspired evaluation: BERTscore scoring using longest common substring

**System Support:**

- SLURM workload manager
- Local model training
