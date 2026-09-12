# NeuroMed Scan AI 🧠

**Explainable multimodal brain-imaging model for early cognitive-risk pattern detection.**

## About This Project

NeuroMed Scan AI is an interactive Streamlit dashboard that uses machine learning to analyze brain imaging data and predict cognitive risk patterns. It combines medication burden, cognitive scores, and neuroimaging biomarkers (hippocampal volume, cortical thickness, PET biomarkers) to identify early signs of cognitive decline.

### Key Features

- **Story Mode** - Narrative-driven interface with a magical theme
- **Science Mode** - Traditional scientific dashboard layout
- **Multiple ML Models** - Compares medication-only, imaging-only, and combined models
- **Explainable AI** - Understand which features influenced the prediction
- **Biomarker Visualization** - See how a patient compares to the dataset
- **Demo Data** - Try it out with synthetic data immediately
- **CSV Upload** - Use your own patient data

## How to Use

### Run Locally

1. Install Python (if you don't have it)
2. Open a terminal/command prompt
3. Run these commands:

```bash
pip install -r requirements.txt
streamlit run app.py
```

4. Your app opens in your browser automatically

### Use Online (Streamlit Cloud)

Visit: **[mindscope-ai.streamlit.app](https://mindscope-ai.streamlit.app)**

## Data Format

If uploading your own CSV, include these columns:

- `age` - Patient age
- `cognitive_score` - MMSE, MoCA, or similar score
- `medication_burden` - Number or score of medications
- `hippocampal_volume` - Brain structure volume
- `cortical_thickness` - Brain cortex thickness
- `pet_biomarker` - PET imaging biomarker (SUVR)
- `risk_label` - Target: "Low", "Moderate", or "High"

Column names are flexible - the app tries to match common variations.

## What the Models Do

- **Medication-only Model** - Uses just medication burden
- **Imaging-only Model** - Uses just brain imaging biomarkers
- **Combined Model** - Uses all features together (most accurate)

## Important Disclaimer

⚠️ **Research and education prototype only.** This dashboard does NOT:
- Diagnose dementia or any medical condition
- Provide medical advice
- Recommend medication changes

Always consult a healthcare professional for medical decisions.

## Technologies

- **Streamlit** - Interactive web app framework
- **Scikit-learn** - Machine learning models
- **Pandas** - Data manipulation
- **Matplotlib** - Data visualization

## Authors

Made with 🧠 and 💜

---

For questions, visit [GitHub](https://github.com/rubber-ducky45/Mindscope-AI-streamlit-)
