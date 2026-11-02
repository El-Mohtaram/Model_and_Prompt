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