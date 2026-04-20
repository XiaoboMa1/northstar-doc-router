# doc-router

A local pipeline that converts a folder of `.txt` documents into structured JSON records for downstream routing. Given a directory of text documents, the pipeline:

1. scans recursively, rejecting unreadable documents at a pre-LLM gate
2. call llm to classify the file and extract following a defined schema, plus calculating string match results for cross-reference
3. reconciles the two results through a decision tree
4. validates the model's structured output against the defined schema
5. writes four JSONs: three record views and one run-metadata file.

## File Structure And Responsibility

```
doc-router/
├── app.py                  entry point. Loads .env and config.json, validates
│                           required fields, constructs the OpenAI client and
│                           MetricsCollector, calls pipeline.run, hands records
│                           to file_store, prints a metrics summary on exit.
├── pipeline.py             orchestration. Recursive scan, pre-LLM gate,
│                           sha256 doc_id, tiktoken counting, batch packing,
│                           LLM dispatch, per-document result mapping and schema
│                           check, record assembly.
├── llm_service.py          LLM boundary. Prompt construction with the schema
│                           map and the list of expected doc_ids, retry on
│                           transient status codes, JSON parsing with code-
│                           fence tolerance, pydantic envelope validation,
│                           return per-doc results or a structured failure.
├── classifier.py           pure decision logic. keyword_score and reconcile
│                           only.
├── file_store.py           persistence. Splits records into the three
│                           view files, assembles runtime_metadata.json, handles
│                           per-file OSError.
├── metrics.py              MetricsCollector dataclass and a one-line show().
├── config.json             runtime policy: paths, keywords, extraction
│                           schemas, reconciliation thresholds, batching
│                           budget, retry parameters.
├── .env.example            OPENAI_API_KEY, MODEL_NAME template.
├── output_template.json    documented example of all four output files.
├── sample_docs/            including one deliberate conflict case between keyword and LLM.
├── output/                 miscellaneous.json, urgent.json,
│                           human_review.json, runtime_metadata.json.
├── design/                 dataflow and implementation specification.
└── tests/                  see VERIFY.md.
```

Each module has one well-defined job, so each failure mode lives in one place: an OpenAI error can only come from `llm_service.py`, a filter rule change only touches `file_store.py`, a decision threshold change only touches `classifier.py` and `config.json`. Tests can target one module at a time with mock inputs.

## Setup And Run

```bash
pip install -r requirements.txt
cp .env.example .env            # set OPENAI_API_KEY and MODEL_NAME
python app.py                   # reads config.json, writes output/
python -m pytest -q             # 57 cases pass in under a second
```

## Downstream Consumption

```python
import json

with open("output/urgent.json")           as f: urgent = json.load(f)
with open("output/human_review.json")     as f: review = json.load(f)
with open("output/runtime_metadata.json") as f: meta   = json.load(f)

# Route view — priority queue ingestion.
for rec in urgent:
    enqueue_priority(rec["doc_id"], rec["extracted_fields"], rec["source_path"])

# Risk view — strict enum dispatch, no string parsing required.
for rec in review:
    reason = rec["classification"]["review_reason"]
    if reason == "conflict":
        audit_conflict(rec)
    elif reason in ("llm_api_error", "llm_envelope_error"):
        page_on_call(rec)
    else:
        enqueue_human_review(rec)

# Metrics — stage-specific counters for dashboards and alerting.
m = meta["metrics"]
assert m["file_processed"] + m["file_errors"] == len(meta["input_file_ids"])
emit("doc_router.llm_api_errors",      m["llm_api_errors"])
emit("doc_router.llm_schema_mismatch", m["llm_schema_mismatch"])
emit("doc_router.llm_missing_docs",    m["llm_missing_docs"])
```

`urgent.json` contains any docs that llm classified as urgent, while `human_review.json` covers docs that were not successfully extracted due to llm api unavailability or extracted but probably wrong due to llm hallucination (e.g., cross-reference with string match results flags a significant conflict). `miscellaneous.json` contains everything else: non-urgent documents whose `review_reason` is null. `runtime_metadata.json` carries the run-level counters and the list of `doc_id`s in the order they were processed.

## Trade-offs And Limitations

1. overlapping records: A document with `route == "urgent"` and a non-null `review_reason` is written into both `urgent.json` and `human_review.json`. Route priority and review risk are separate questions, and maybe required by separate consumers. The cost is duplicated storage and an extra logic to ensure file-related metrics are calculated correctly.

2. file checks and error handling: 
- `metrics.file_errors` granularity: it counts both pre-LLM gate rejections (unreadable input) and output-write failures (disk or permission errors). 
- An oversize document is rejected at the gate and therefore never gets a keyword-only fallback route, even though the keyword score could in principle be computed from the text the pipeline already read. 
- Whitespace-only content (`"   \n"`) is treated as valid input and goes to the LLM, because "empty" is defined as zero decoded characters, not zero non-whitespace characters. 

3. strict enum values for `error_reason` and `review_reason`: Consumers can filter with `==` rather than substring search. The cost is that HTTP status codes, pydantic validation messages, and truncated-JSON hints never appear in the record itself; the next step for PoC is to write them into a logger.

4. llm interface:
- OpenAI support only
- retry behaviour has two rough edges: OpenAI SDK raises `RateLimitError` for 429, but `llm_service._invoke` treats that exception as retryable unconditionally before it looks at the status list. Separately, `Retry-After` headers from the server are ignored; the backoff is a fixed `retry_backoff_base_seconds * 2 ** attempt`; a production version would read `Retry-After` and make 429 handling opt-out through config.
