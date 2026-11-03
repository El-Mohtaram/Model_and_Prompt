# README – Final Data

## 1. Folder Contents

The `Final Data` folder contains the generated synthetic dataset. It includes two main types of data:

- **Candidates:** Synthetic candidate profiles containing skills, CV information, functional abilities, assistive technology needs, and preferences.
- **Jobs:** Synthetic job records containing job descriptions, required skills, job demands, and workplace requirements.


## 2. Files to Use

To work with the generated data, the following folders and files should be referenced:

| File | Purpose |
|---|---|
| `Final Data/candidates` | Contains the generated candidate records. |
| `Final Data/jobs` | Contains the generated job postings. |
| `schemas/job_m2.json` | Defines the structure and validation rules for job records. |
| `schemas/candidate_v0_2.json` | Defines the structure and validation rules for candidate records. |
| `registry/registry.yaml` | Explains the item codes used for candidate capabilities, job demands, and workplace requirements. |
| `registry/job_templates.yaml` | Contains occupation templates and skill references used during generation. |

**The most important supporting file is `registry/registry.yaml`.** It provides the definitions needed to interpret the structured codes in the candidate and job records. Do not assume the meaning of an item code without checking the registry.

## 3. Understanding Candidate Records

Each candidate record contains the following information:

- `candidate_id`: Unique candidate identifier.
- `job_category`: Candidate category (`mobility`, `office`, or `both`).
- `disability`: Human-readable disability-related description. This is included because M1 uses it to extract relevant disability-related terms and phrases.
- `skills`: Candidate skills, each with a proficiency level from 1 to 5.
- `cv_text`: Professional background information, such as experience, projects, and awards.
- `capabilities`: Structured information describing the candidate's functional abilities.
- `input_methods`: Methods the candidate can use to interact with devices and systems.
- `usable_formats`: Accessible formats the candidate can use.
- `communication_channels`: Available communication methods.
- `at_needs`: Assistive technology requirements and accommodation needs.
- `preferences`: Additional employment-related preferences.

Candidate records use schema version `0.2`.

### Important notes

- Skills are free-text labels, so similar skills may appear with different wording.
- Capability values may be numerical, scores from 1 to 5, or `null`, depending on the item.
- A `null` value means that information is unavailable or unspecified.
- CV information may be incomplete. 
- The `job_category` field helps organize candidates but should not be treated as a final decision about job suitability.

## 4. Understanding Job Records

Job records describe synthetic employment opportunities and include information about:

- Job identifiers and categories.
- Job descriptions and required skills.
- Job demands.
- Workplace requirements and conditions.
- Other job-related information used for matching.

Job demand and workplace codes must be interpreted using `registry/registry.yaml`.

In particular:
- `D01–D17`: Job demand items.
- `E09–E27`: Workplace-related items.

Refer to the registry for the exact definition of each item.

## 5. Generation and Evaluation Files

The following files are useful for understanding how the dataset was generated. They are not normally required as inputs to the matching system.

| File | Purpose |
|---|---|
| `registry/config.yaml` | Contains generation settings, including the random seed, generation weights, and candidate/job counts. |
| `generator/candidates.py` | Contains the candidate generation logic. |
| `generator/jobs.py` | Contains the job generation logic. |
| `generator/ability_model.py` | Contains the logic used to generate candidate ability values and scores. |
| `manifest.json` | Provides metadata about the generated dataset, such as record counts and generation settings, if included. |


## 6. Recommended Workflow

1. Load the candidate and job records from `Final Data`.
2. Validate each record using its corresponding JSON schema.
3. Refer to `registry/registry.yaml` to understand the structured item codes.
4. Extract candidate skills, CV information, capabilities, and assistive technology needs.
5. Extract job skills, demands, and workplace requirements.

## 7. Dataset Limitations

- All candidates and jobs are synthetic; they do not represent real individuals or actual vacancies.
- Generation weights and distributions are design assumptions, not real-world population statistics.
- Some records may contain missing or incomplete information.
- Similar skills may be expressed using different wording.
- Results obtained using this dataset do not necessarily generalize to real-world candidates and job postings.

The matching process should focus on functional abilities, skills, assistive technology needs, and workplace compatibility rather than using disability descriptions or diagnoses alone to determine job suitability.

## 8. Qwen Capability-Demand Workflow

The active Qwen workflow uses only two input sections:

- `candidate_capabilities`
- `job_demands`

The Qwen service is implemented in `qwen-for-assistive.ipynb`. It exposes `/health`, `/chat`, and the structured `/match` endpoint. The local client is in `llm/kaggle_client.py`; it rejects unrelated fields before sending a request.

The Kaggle notebook is the inference server. Keep it running while the local M3 notebook or client calls the current Cloudflare URL. The Kaggle secret label is `SYNTHETICDATA_API_TOKEN`; the local `KAGGLE_LLM_API_TOKEN` must contain the same application-token value. These are not Kaggle account API credentials.

The current prompt is versioned as `match-system-v1` and its source is stored in `prompts/match_system_v1.txt`, with synthetic demonstrations in `prompts/match_examples_v1.json`. Qwen must return structured `Met`, `Met with accommodation`, `Gap`, or `Unknown` results with evidence and confidence.

Recommended smoke-test order:

1. Start the Kaggle notebook and verify local and public `/health`.
2. Verify local and public `/match` with the one-pair PC06/D02 test.
3. Configure the local client with the current tunnel URL and application token.
4. Run one real capability-demand pair.
5. Evaluate prompt versions on a separate labelled dataset before enabling broader experiments.

The M3 notebook remains an internal deterministic reference and regression fixture. It is not the reported Qwen approach. The legacy `tests/build_ready_notebook.py` script regenerates the older M3 reference notebook and should not be used to update the active Qwen workflow without synchronizing it first.

## 9. Final Model Selection

The repository uses two model roles, kept separate:

- **Qwen GGUF on Kaggle:** capability-demand interpretation and structured JSON generation. A small local model was selected because it can run on a Kaggle GPU, keeps inference cost low, and can be constrained by a narrow prompt and response schema. The current model choice is an engineering baseline, not a claim that it is universally optimal; it should be compared with alternatives using the same held-out evaluation cases.
- **all-MiniLM-L6-v2 with SBERT/Chroma:** semantic skill retrieval in the companion `MatchingModule-main/MatchingModule-main/m3_nlp` module. This module is separate from the reported Qwen capability-demand workflow.

The final reported approach is the Qwen capability-demand path. The deterministic M3 notebook is retained for internal regression checks only and is not reported as the matching method.

## 10. Capability-Demand Trigger and Skill Matching

The active Qwen request is intentionally limited to `candidate_capabilities` and `job_demands`. This determines whether the capability-demand analysis is applicable.

Skill matching is a separate branch. The intended orchestration rule is:

1. If the task contains capability and demand records, call Qwen `/match` with only those sections.
2. If the task explicitly requests semantic skill matching and both records contain skills, invoke the companion MatchingModule.
3. Do not send skills to the Qwen capability-demand prompt.
4. Do not infer a capability from a skill, disability, CV, or diagnosis field.

The current repository exposes the Qwen endpoint and preserves the MatchingModule as a separate module; a production orchestrator still needs to implement this branch-selection trigger explicitly.

### MatchingModule summary

The companion module is an NLP semantic candidate-job matcher. According to its README, it:

- normalizes skill and text fields conservatively;
- builds context-aware candidate and job representations;
- fine-tunes `sentence-transformers/all-MiniLM-L6-v2`;
- creates Jaccard-derived weak labels with negative sampling and candidate-disjoint train/validation splits;
- evaluates with validation loss, Pearson correlation, Spearman correlation, and MSE;
- stores job embeddings in persistent ChromaDB;
- retrieves ranked jobs using cosine distance and the explicit contract `cosine_similarity = 1 - cosine_distance`;
- writes ranked recommendations and processed audit artifacts.

Its processed scores are weak-label or retrieval artifacts unless an independently reviewed ground-truth set is supplied. They should not automatically be described as human-validated truth.

## 11. Prompt Engineering Choices

The current prompt is versioned as `match-system-v1` in `prompts/match_system_v1.txt`, with controlled demonstrations in `prompts/match_examples_v1.json`.

The design uses:

- a narrow role and input allowlist;
- explicit negative instructions against protected-field inference;
- structured few-shot examples for Met, Gap, and Unknown cases;
- explicit treatment of missing evidence;
- strict output labels and evidence fields;
- deterministic decoding settings where supported;
- response parsing and schema validation after generation;
- prompt versioning so later prompt changes can be compared.

The prompt is not fine-tuning and does not permanently teach the model the dataset. Demonstrations condition the current request only. Prompt quality must be measured on separate development, validation, and held-out test cases.

## 12. Fairness, Bias, and Evaluation Literature

Suggested references supporting the design and evaluation choices:

- Yang et al., *Qwen2.5 Technical Report* (2024): model capabilities and limitations; use the model card for the exact deployed checkpoint.
- Liang et al., *Holistic Evaluation of Language Models* (TMLR, 2023): evaluate accuracy, robustness, fairness, and efficiency together.
- Mitchell et al., *Model Cards for Model Reporting* (FAccT, 2019): document intended use, limitations, and evaluation conditions.
- Hoffmann et al., *Training Compute-Optimal Large Language Models* (2022): model-size, data, and compute tradeoffs.
- Brown et al., *Language Models are Few-Shot Learners* (NeurIPS, 2020): few-shot demonstrations as task conditioning.
- Reynolds and McDonell, *Prompt Programming for Large Language Models* (2021): prompts as task specifications.
- Liu et al., *Pre-train, Prompt, and Predict: A Systematic Survey of Prompting Methods* (ACM Computing Surveys, 2023): prompt-method taxonomy.
- Suresh and Guttag, *A Framework for Understanding Sources of Harm throughout the Machine Learning Life Cycle* (EAAMO, 2021): data, measurement, aggregation, and deployment harms.
- Mehrabi et al., *A Survey on Bias and Fairness in Machine Learning* (ACM Computing Surveys, 2021): fairness terminology and measurement background.
- Selbst et al., *Fairness and Abstraction in Sociotechnical Systems* (FAccT, 2019): why employment matching must be treated as a sociotechnical system.
- NIST, *AI Risk Management Framework 1.0* (2023): governance, transparency, measurement, and risk controls.

These references justify the engineering choices but do not establish model quality. Any final performance or fairness claim requires an independent evaluation set and documented results.