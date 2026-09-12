
# NeuroMed Scan AI — Streamlit Starter

This starter dashboard includes:

- Story Mode and Science Mode
- Demo data for 80 participants
- CSV upload support
- Medication-only, imaging-only, and combined models
- Accuracy, precision, recall, F1 score
- Feature importance
- Confusion matrix
- Risk pattern, AI confidence, and explanation cards
- Blurred cards while loading
- A placeholder telescope/constellation loading sequence
- A simple firefly-and-flower ending made with CSS
- A permanent safety note

## Run it on Windows

Open Command Prompt or PowerShell inside this folder:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Streamlit will open the dashboard in your browser.

## Required CSV columns

The app accepts several common column names, but the easiest names are:

- `participant_id`
- `age`
- `cognitive_score`
- `medication_burden`
- `hippocampal_volume`
- `cortical_thickness`
- `pet_biomarker`
- `risk_label`

The risk label is the target the model predicts. Do not include it as an input feature.

## Add your approved assets later

Put approved PNG files in `assets/` and use these names:

- `twilight_meadow.png`
- `night_meadow.png`
- `midnight_meadow.png`
- `bunnies_telescope.png`
- `bunnies_happy_dance.png`
- `bunnies_flower_toss.png`
- `bunnies_sitting_cards.png`
- `bunnies_walking.png`
- `house_door_sequence.png`
- `risk_low.png`
- `risk_moderate.png`
- `risk_high.png`

The starter already works without the assets. The CSS placeholders can be replaced later.

## Important science-fair note

The included demo dataset is synthetic. It is only for testing the dashboard code. Replace it with your cleaned, de-identified research dataset before reporting project results.

The dashboard is a research and education prototype. It is not a diagnostic tool.
