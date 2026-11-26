# ART Summary Workbench, version 2

Research prototype for a master's dissertation. All data here is made up. Not for clinical use.

## Run
    pip install -r requirements.txt
    streamlit run streamlit_app.py
    python tests/test_all.py

## Tabs
| Tab | Chapter 3 | Saves to |
|---|---|---|
| 1. Essential features | 3.8 panel, 4 point scale, I-CVI | study/feature_ratings.csv |
| 2. Summariser | 3.9, 3.11.4 baselines, 3.12.2 audit | |
| Gold summaries | 3.10 | study/gold_summaries.csv |
| 3a. Clinician ratings | 3.12.3 blind ratings and harm | study/ratings.csv |
| 3b. Review time | 3.12.4 crossover and 8 questions | study/reviews.csv |
| Results | 3.12 analysis and acceptance criteria | |
| Test a file | upload made-up notes | |

- test_data/art_dummy_data_100.csv: 100 made-up records in the same wording as the 40 record file.

## BART
Not trained. Open train_on_colab.ipynb in Google Colab (synthetic data only), or run train_bart.py on a GPU machine.
Put the test predictions CSV in the predictions folder and the app shows them.

## Test files
- test_data/art_test_dataset.csv: 40 made-up records in wording the rules were built for. The reader scored 100%.
  This is a software check and says nothing about real notes.
- test_data/art_hard_notes.csv: 6 notes in clinic shorthand the rules were not built for. The reader misses
  several facts (TB screen and next appointment 3 of 6 correct).

## Still to do
- De-identification pipeline (3.7)
- Fine-tuned BART, BioBART and zero-shot results
- Clinician-resolved abbreviation list (3.9.2)
- Record sets matched on care stage and length (3.12.4)
- WHO stage and viral load history in the coded block. Longformer check (3.9.3)
- Separate logins and encrypted storage. Anyone using the app can open every tab, including the scorer panel
- Confirm the suppression threshold (set to below 1000 copies/mL in facts.py)
- Final kappa on the 30 summary audit subset and the human ceiling for key fields

