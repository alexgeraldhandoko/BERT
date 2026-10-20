# Source material and where it is used

| Material | Source | Placement in this repository |
| --- | --- | --- |
| SQuAD paper | Rajpurkar, Zhang, Lopyrev, and Liang (2016), [*SQuAD: 100,000+ Questions for Machine Comprehension of Text*](https://aclanthology.org/D16-1264.pdf), EMNLP, pp. 2383–2392. | Cited here and used to describe the extractive answer-span task. The paper PDF is not copied into the repo. |
| SQuAD 1.1 training and dev data | Course Canvas Project folder: `train-v1.1.json` and `dev-v1.1.json`. | Local copies in `data/raw/`, ignored by Git. The code reads these paths; teammates obtain their own copies from Canvas. |
| Official course scorer | Course-provided `evaluate-v2.0.py`, linked at [CodaLab](https://worksheets.codalab.org/rest/bundles/0x6b567e1cf2e041ec80d7098f031c5c9e/contents/blob/) and available in the course Project folder. The [SQuAD authors' repository](https://github.com/rajpurkar/SQuAD-explorer/blob/master/evaluate-v2.0.py) also contains this scorer. | Exact local course file copied to `vendor/evaluate-v2.0.py`; license and provenance are in `vendor/`. `src/qa_baseline/evaluate.py` invokes it without reimplementing EM/F1. |
| Group proposal | CS4248 Project Proposal PDF in the course Project folder. | Informs the general BERT QA starting point. Its calibration method is optional branch work and is not built into this common scaffold. |
| Course project requirements | `cs4248_project_requirements.pdf` in the course Project folder. | README includes run instructions; this file distinguishes our baseline code from the external scorer and model libraries, as required for the final submission. |

The SQuAD paper describes answers as spans of passage text. This is why the baseline uses a start/end span head and keeps tokenizer character offsets to turn token positions back into the original passage text. The course specifies the SQuAD 2.0 scorer with SQuAD 1.1 files; the checked-in scorer is the course copy and is exercised with an answerable-only 1.1 fixture.

