# Technical Assessment: Stabilise a Fragile Document Processor

Estimated time: 90-120 minutes

## Background

You have inherited a small internal Python prototype that loops through text documents, sends them to an LLM, assigns a rough route such as invoice or complaint, and writes local output files.

The prototype was rushed, and the current code does not meet normal engineering standards.

You have been given the starter code as-is.

To make the task concrete, a small `sample_docs/` folder is included for local testing and demonstration.

## The Task

Improve the starter into a small, credible local project.

You should make it safer, easier to run, and easier to understand while keeping the solution proportionate to the scope.

Think of this as a miniature version of the kind of reusable AI-enabled component work you might do in a central AI starter-kit team.

## What The Starter Includes

- `app.py`
- `processor.py`
- `llm_service.py`
- `routes.py`
- `file_store.py`
- `metrics.py`
- `.env.example`
- `config.json`
- `sample_docs/`

The code intentionally contains a mixture of poor practices, brittle assumptions, and missing project basics.

Assume this starter reflects a rushed pilot that now needs to be made safe enough for another engineer to run and extend.

## What Your Solution Should Do

Your submission should improve the project so that it:

- can be run locally with clear setup steps
- does not keep secrets or machine-specific paths in source code
- handles model-call failures in a visible and sensible way
- is easier to read and maintain than the starter
- includes basic tests
- produces a structured output from the sample documents
- includes a lightweight way to verify the output or routing behavior

## Suggested Scope

You do not need to build a full production system.

A strong solution will usually:

- separate config from code
- make the processor reusable and testable
- keep the LLM boundary small and mockable
- generate a clear local output format such as JSON
- include a few focused tests or an equivalent verification step
- document how someone else would run and extend it

## Handover Artifact

As part of your handover, please provide:

- your improved implementation
- a short README or runbook, approximately one page
- a minimal dependency file
- a few focused tests
- the structured output produced from `sample_docs/`
- a short note on how you verified the behavior

## What We Are Assessing

We are looking for:

- pragmatism
- engineering hygiene
- reliability thinking
- ability to work cleanly with APIs and data
- a lightweight evaluation mindset
- clarity of handover
- sensible use of AI-assisted development

## Submission

Please provide:

- your implementation
- your README or runbook
- your tests
- your output artifact

## Notes

- This is a proof-of-concept exercise, not a production build.
- You do not need to introduce extra infrastructure unless you believe it is justified.
- Clear reasoning and sensible tradeoffs matter more than complexity.
- If you make assumptions, state them clearly.
- You may mock the model call in tests if that helps you keep the solution deterministic.
