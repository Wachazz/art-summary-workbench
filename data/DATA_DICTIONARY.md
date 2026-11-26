# ART Summary Workbench: synthetic dataset

SYNTHETIC DATA ONLY. Every patient, date and value is invented by generate_data.py (seed 2026). No real patient record is used. Not for clinical use. Replace with ethics-approved records (MRCZ) before reporting findings.

## Files
- art_synthetic_records.csv: one row per patient (500 rows). The first 30 match the records inside the app.
- art_synthetic_visits.csv: one row per clinic visit (3,034 rows), linked by record_id.

## Columns in art_synthetic_records.csv
| Column | Meaning |
|---|---|
| record_id | Synthetic ID, ART-1000 to ART-1499 |
| split | train (400), validation (50) or test (50). Split by patient, so no patient appears in two sets |
| age | Age in years, 19 to 62 |
| sex | F or M |
| art_start_date | Date ART started |
| n_visits | Number of visits, 4 to 8 |
| last_visit_date | Date of the final visit |
| current_regimen | Regimen at the last visit |
| regimen_changed_from | Earlier regimen if it was switched, otherwise none |
| viral_load_copies_ml | Latest viral load, 0 means not detected |
| viral_load_status | suppressed (0, 40, 150, 380) or not suppressed (12000) |
| cd4_cells_mm3 | Latest CD4 count |
| adherence | good or poor, from pill count |
| tb_screen | negative or positive |
| allergy | none, cotrimoxazole rash or penicillin allergy |
| comorbidity | none, hypertension, type 2 diabetes or asthma |
| next_appointment | Date of the next appointment |
| word_count_source | Words in source_text |
| source_text | All visit notes joined, each starting with its [date]. Model input |
| reference_summary | Stand-in for a clinician written gold summary. Model target |

## Columns in art_synthetic_visits.csv
record_id, split, visit_number, visit_date, note_text.

## Limits
- Reference summaries are rule-written stand-ins, not written by clinicians.
- Notes are short and use a small set of filler sentences, so real records will be harder.
- Category rates (for example TB screen positive about 27 percent) come from the generator and do not match real clinic rates.
