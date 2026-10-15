---
name: prepare-viva
description: Prepare evidence-grounded viva questions and answers from the implemented Fakeddit project.
argument-hint: "[topic or project component]"
agent: Research Reviewer
---

Prepare viva material for: ${input:topic:topic or project component}

Inspect the current source, experiment result files, model artifacts, and project status documentation. Generate likely examiner questions with concise, technically accurate answers based only on inspectable implementation and recorded measurements.

- Cite the relevant repository files/results for each substantive answer.
- Explicitly label features as completed, partial, or planned.
- State subset sizes, split protocol, measured metrics, limitations, and failures exactly as recorded.
- If evidence is absent or contradictory, say so and propose an honest answer; do not fill gaps with assumptions.
- Do not add unsupported literature, novelty, performance, or deployment claims.
