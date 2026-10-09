# Datasets
`processed/{task}_{train,test}.jsonl` (2,000 / 200 rows; fields `prompt`, `target`, `answer`) for the tasks
gsm8k, arith, logic, mt, summ, emotion. Regenerate with `python src/prep_data.py` from the workspace root
(downloads from the Hugging Face Hub; sources are listed in `../resources.md`).
