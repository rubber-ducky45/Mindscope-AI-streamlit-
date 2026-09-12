
from __future__ import annotations

import base64
import io
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split


# ------------------------------------------------------------
# PAGE SETUP
# ------------------------------------------------------------
st.set_page_config(
    page_title="NeuroMed Scan AI",
    page_icon="🧠",
    layout="wide",
)

APP_DIR = Path(__file__).parent
ASSET_DIR = APP_DIR / "assets"


def find_asset(
    filenames: list[str],
    required_words: tuple[str, ...] = (),
) -> Path | None:
    """Find an approved image anywhere inside the project's assets folder."""
    if not ASSET_DIR.exists():
        return None

    png_files = [
        path
        for path in ASSET_DIR.rglob("*.png")
        if path.is_file()
    ]

    by_name = {path.name.lower(): path for path in png_files}

    for filename in filenames:
        match = by_name.get(filename.lower())
        if match is not None:
            return match

    if required_words:
        lowered_words = tuple(word.lower() for word in required_words)

        for path in png_files:
            searchable = path.stem.lower().replace("-", "_").replace(" ", "_")

            if all(word in searchable for word in lowered_words):
                return path

    return None


@st.cache_data(show_spinner=False)
def asset_data_uri(path_string: str | None) -> str:
    """Convert a local PNG into a browser-safe CSS data URI."""
    if not path_string:
        return ""

    path = Path(path_string)

    if not path.exists():
        return ""

    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


APPROVED_ASSETS = {
    "upload_button": find_asset(["data_button.png"]),
    "analysis_button": find_asset(["analysis_button.png"]),
    "results_button": find_asset(["results_button.png"]),
    "stars": find_asset(["color_stars.png"]),
    "comet": find_asset(["comet.png"]),
    "twilight_meadow": find_asset(
        [
            "twilight_meadow.png",
            "magical_twilight_meadow_with_glowing_mushrooms.png",
            "gentle_twilight_meadow_with_glowing_mushrooms.png",
        ],
        required_words=("twilight", "meadow"),
    ),
    "night_meadow": find_asset(
        [
            "night_meadow.png",
            "whimsical_moonlit_meadow_at_night.png",
            "moonlit_meadow_with_glowing_mushrooms.png",
        ],
        required_words=("night", "meadow"),
    ),
    "midnight_meadow": find_asset(
        [
            "midnight_meadow.png",
            "magical_moonlit_meadow_with_glowing_mushrooms.png",
        ],
        required_words=("midnight", "meadow"),
    ),
}


# ------------------------------------------------------------
# STYLING
# ------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Balsamiq+Sans:wght@400;700&family=Montserrat:wght@400;500;600;700;800&family=Quicksand:wght@400;500;600;700&display=swap');

    html,
    body,
    [data-testid="stAppViewContainer"],
    [data-testid="stAppViewContainer"] > .main,
    .stApp {
        height: 100vh !important;
        overflow: hidden !important;
    }

    [data-testid="stSidebar"] {
        overflow-y: auto !important;
    }

    .main .block-container {
        height: 100vh !important;
        max-height: 100vh !important;
        overflow: hidden !important;
        padding-top: .45rem !important;
        padding-bottom: .45rem !important;
    }

    [data-testid="stTabs"] {
        height: calc(100vh - 260px);
        overflow: hidden;
    }

    [data-testid="stTabs"] [data-baseweb="tab-panel"] {
        height: calc(100vh - 310px);
        overflow: hidden;
        padding-top: .45rem;
    }

    .title-card {
        padding: .72rem 1rem !important;
        margin-bottom: .45rem !important;
        border-radius: 20px !important;
    }

    .title-card h1 {
        font-size: 1.9rem !important;
    }

    .title-card p {
        font-size: .9rem !important;
    }

    .safety-note {
        padding: .55rem .8rem !important;
        margin: .25rem 0 .55rem !important;
        font-size: .86rem !important;
    }

    .story-strip {
        padding: .55rem .75rem !important;
        margin: .25rem 0 .55rem !important;
    }

    .result-card {
        min-height: 118px !important;
        padding: .72rem .85rem !important;
    }

    .result-card h3 {
        font-size: 1.02rem !important;
        margin-bottom: .35rem !important;
    }

    .big-number {
        font-size: 1.55rem !important;
    }

    .risk-flower {
        font-size: 2.7rem !important;
    }

    .small-muted {
        font-size: .78rem !important;
    }

    .compact-panel {
        height: calc(100vh - 330px);
        overflow: hidden;
    }

    .compact-scroll-panel {
        height: calc(100vh - 330px);
        overflow-y: auto;
        padding-right: .35rem;
    }

    :root {
        --brown: #7B563E;
        --cream: #FFF9F2;
        --lavender: #EDE6FF;
        --teal: #A7DED8;
        --yellow: #F8D77B;
        --pink: #F6A7B8;
        --ink: #3F3348;
    }

    .stApp {
        background:
            radial-gradient(circle at 15% 10%, rgba(255,255,255,.62), transparent 28%),
            linear-gradient(180deg, #B7A1EF 0%, #8D83D8 32%, #6F78B8 65%, #516E78 100%);
        color: var(--ink);
    }

    .main .block-container {
        padding-top: 1.1rem;
        padding-bottom: 4rem;
        max-width: 1320px;
    }

    .title-card {
        border: 1.5px solid rgba(123, 86, 62, .72);
        border-radius: 26px;
        padding: 1.2rem 1.4rem;
        background: rgba(255, 249, 242, .72);
        backdrop-filter: blur(12px);
        box-shadow: 0 12px 28px rgba(54, 40, 70, .16);
        margin-bottom: 1rem;
    }

    .title-card h1 {
        margin: 0;
        color: #4F3D59;
        letter-spacing: .02em;
    }

    .title-card p {
        margin: .35rem 0 0;
        color: #66546D;
    }

    .safety-note {
        opacity: .72;
        transition: opacity .25s ease, transform .25s ease;
        border: 1.5px solid rgba(123, 86, 62, .75);
        border-radius: 18px;
        padding: .8rem 1rem;
        background: rgba(218, 246, 228, .88);
        color: #5B4334;
        margin: .5rem 0 1rem;
        box-shadow: 0 8px 20px rgba(55, 40, 66, .10);
    }

    .safety-note:hover {
        opacity: 1;
        transform: translateY(-2px);
    }

    .result-card {
        border: 1.5px solid rgba(123, 86, 62, .72);
        border-radius: 22px;
        padding: 1rem 1.1rem;
        min-height: 150px;
        background: rgba(255, 250, 246, .68);
        backdrop-filter: blur(16px);
        box-shadow:
            0 10px 28px rgba(55, 40, 70, .15),
            inset 0 1px 0 rgba(255,255,255,.70);
        transition: filter .9s ease, opacity .9s ease, transform .9s ease;
    }

    .result-card.loading {
        filter: blur(8px);
        opacity: .42;
        transform: scale(.985);
    }

    .result-card.ready {
        filter: blur(0);
        opacity: 1;
        transform: scale(1);
    }

    .result-card h3 {
        margin-top: 0;
        color: #5B4334;
    }

    .big-number {
        font-size: 2.15rem;
        font-weight: 800;
        color: #5B4334;
        line-height: 1.05;
    }

    .small-muted {
        color: #6F6476;
        font-size: .92rem;
    }

    .risk-flower {
        font-size: 4rem;
        line-height: 1;
        filter: drop-shadow(0 5px 7px rgba(75,54,64,.16));
    }

    .story-strip {
        border: 1.5px solid rgba(123, 86, 62, .55);
        border-radius: 20px;
        background: rgba(255,255,255,.48);
        padding: .9rem 1rem;
        margin: .6rem 0 1rem;
        backdrop-filter: blur(10px);
    }

    .wood-button-hint {
        text-align: center;
        color: #6C4C38;
        font-size: .88rem;
        margin-top: -.3rem;
    }

    .firefly-field {
        position: relative;
        height: 220px;
        overflow: hidden;
        border-radius: 24px;
        border: 1.5px solid rgba(123, 86, 62, .55);
        background: linear-gradient(180deg, rgba(15,26,73,.82), rgba(28,62,72,.74));
    }

    .firefly {
        position: absolute;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #FFE66D;
        box-shadow: 0 0 8px #FFE66D, 0 0 18px #FFD43B;
        animation: drift 5s ease-in-out infinite alternate;
    }

    @keyframes drift {
        from { transform: translate(-8px, 4px) scale(.7); opacity: .45; }
        to   { transform: translate(18px, -18px) scale(1.15); opacity: 1; }
    }

    .flower-row {
        position: absolute;
        bottom: 14px;
        left: 0;
        right: 0;
        text-align: center;
        font-size: 2.1rem;
        letter-spacing: .7rem;
    }


    .notebook-card {
        font-family: 'Balsamiq Sans', cursive !important;
        background:
            repeating-linear-gradient(
                to bottom,
                rgba(255, 253, 239, .96) 0px,
                rgba(255, 253, 239, .96) 29px,
                rgba(174, 205, 219, .40) 30px
            ) !important;
        border-left: 5px solid rgba(237, 152, 165, .72) !important;
        color: #514252;
    }

    .notebook-card,
    .notebook-card h3,
    .notebook-card p,
    .notebook-card strong,
    .notebook-card span {
        font-family: 'Balsamiq Sans', cursive !important;
    }

    div.stButton > button {
        border-radius: 16px;
        border: 1.5px solid #8B6042;
        background: linear-gradient(180deg, #D9A76A, #B97B46);
        color: #FFF9F1;
        font-weight: 800;
        box-shadow: 0 7px 0 #7A4F32, 0 10px 20px rgba(68,44,35,.18);
        transition: transform .15s ease, box-shadow .15s ease, filter .15s ease;
    }

    div.stButton > button:hover {
        transform: translateY(-2px);
        filter: brightness(1.06);
        box-shadow: 0 9px 0 #7A4F32, 0 12px 24px rgba(68,44,35,.22);
    }

    div.stButton > button:active {
        transform: translateY(4px);
        box-shadow: 0 3px 0 #7A4F32, 0 6px 14px rgba(68,44,35,.18);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------
COLUMN_ALIASES = {
    "participant_id": ["participant_id", "subject_id", "rid", "ptid", "id"],
    "age": ["age", "age_years"],
    "cognitive_score": [
        "cognitive_score",
        "mmse",
        "moca",
        "adas13",
        "adas_cog",
        "cognitive",
    ],
    "medication_burden": [
        "medication_burden",
        "medication_burden_score",
        "med_burden",
        "anticholinergic_burden",
    ],
    "hippocampal_volume": [
        "hippocampal_volume",
        "hippocampus_volume",
        "hippocampal_vol",
        "hippocampus",
    ],
    "cortical_thickness": [
        "cortical_thickness",
        "mean_cortical_thickness",
        "cortex_thickness",
    ],
    "pet_biomarker": [
        "pet_biomarker",
        "amyloid_suvr",
        "centiloid",
        "pet_suvr",
        "fdg_suvr",
        "tau_suvr",
    ],
    "risk_label": ["risk_label", "risk", "target", "class", "diagnosis_group"],
}


def normalize_name(name: str) -> str:
    return (
        str(name)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
        .replace("(", "")
        .replace(")", "")
    )


def match_columns(df: pd.DataFrame) -> dict[str, str | None]:
    normalized = {normalize_name(c): c for c in df.columns}
    matched: dict[str, str | None] = {}

    for canonical, aliases in COLUMN_ALIASES.items():
        found = None
        for alias in aliases:
            if alias in normalized:
                found = normalized[alias]
                break
        matched[canonical] = found
    return matched


@st.cache_data
def make_demo_data() -> pd.DataFrame:
    rng = np.random.default_rng(42)

    age_groups = [(50, 59), (60, 69), (70, 79), (80, 89)]
    ages = np.concatenate(
        [rng.integers(low, high + 1, size=20) for low, high in age_groups]
    )

    cognitive = np.clip(31 - 0.16 * (ages - 50) + rng.normal(0, 2.1, 80), 10, 30)
    med_burden = np.clip(
        np.round(0.03 * (ages - 50) + rng.normal(1.2, 1.0, 80)), 0, 6
    )
    hippocampal = np.clip(
        7600 - 46 * (ages - 50) + rng.normal(0, 420, 80), 3600, 8200
    )
    cortical = np.clip(
        2.72 - 0.0085 * (ages - 50) + rng.normal(0, 0.07, 80), 2.0, 2.85
    )
    pet = np.clip(
        0.82 + 0.012 * (ages - 50) + rng.normal(0, 0.12, 80), 0.55, 1.65
    )

    hidden_score = (
        0.045 * (ages - 50)
        + 0.62 * med_burden
        - 0.25 * (cognitive - 20)
        - 0.00075 * (hippocampal - 5500)
        - 2.4 * (cortical - 2.35)
        + 2.7 * (pet - 1.0)
        + rng.normal(0, 0.8, 80)
    )

    q1, q2 = np.quantile(hidden_score, [0.34, 0.67])
    labels = np.where(
        hidden_score <= q1,
        "Low",
        np.where(hidden_score <= q2, "Moderate", "High"),
    )

    age_group = pd.cut(
        ages,
        bins=[49, 59, 69, 79, 89],
        labels=["50–59", "60–69", "70–79", "80–89"],
        include_lowest=True,
    )

    return pd.DataFrame(
        {
            "participant_id": [f"NM_{i+1:03d}" for i in range(80)],
            "age": ages,
            "age_group": age_group.astype(str),
            "cognitive_score": np.round(cognitive, 1),
            "medication_burden": med_burden.astype(int),
            "hippocampal_volume": np.round(hippocampal, 0),
            "cortical_thickness": np.round(cortical, 3),
            "pet_biomarker": np.round(pet, 3),
            "risk_label": labels,
        }
    )


def prepare_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    matched = match_columns(df)

    required = [
        "age",
        "cognitive_score",
        "medication_burden",
        "hippocampal_volume",
        "cortical_thickness",
        "pet_biomarker",
        "risk_label",
    ]
    missing = [key for key in required if matched.get(key) is None]
    if missing:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing)
            + ". You can rename your columns or use the demo dataset."
        )

    out = pd.DataFrame()
    for canonical in required:
        out[canonical] = df[matched[canonical]]

    participant_col = matched.get("participant_id")
    if participant_col is not None:
        out["participant_id"] = df[participant_col].astype(str)
    else:
        out["participant_id"] = [f"Row_{i+1}" for i in range(len(out))]

    for col in [
        "age",
        "cognitive_score",
        "medication_burden",
        "hippocampal_volume",
        "cortical_thickness",
        "pet_biomarker",
    ]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out["risk_label"] = out["risk_label"].astype(str).str.strip().str.title()
    out = out.dropna().reset_index(drop=True)

    if out["risk_label"].nunique() < 2:
        raise ValueError("The risk_label column needs at least two different classes.")

    return out, {k: v for k, v in matched.items() if v is not None}


def train_models(df: pd.DataFrame):
    medication_features = ["medication_burden"]
    imaging_features = [
        "hippocampal_volume",
        "cortical_thickness",
        "pet_biomarker",
    ]
    combined_features = [
        "age",
        "cognitive_score",
        "medication_burden",
        "hippocampal_volume",
        "cortical_thickness",
        "pet_biomarker",
    ]

    X = df[combined_features]
    y = df["risk_label"]

    test_size = 0.25 if len(df) >= 40 else 0.30

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=42,
        stratify=y,
    )

    model_specs = {
        "Medication-only": medication_features,
        "Imaging-only": imaging_features,
        "Combined": combined_features,
    }

    models = {}
    rows = []
    predictions = {}

    for name, features in model_specs.items():
        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=6,
            min_samples_leaf=2,
            random_state=42,
            class_weight="balanced",
        )
        model.fit(X_train[features], y_train)
        pred = model.predict(X_test[features])

        models[name] = (model, features)
        predictions[name] = pred
        rows.append(
            {
                "Model": name,
                "Accuracy": accuracy_score(y_test, pred),
                "Balanced Accuracy": balanced_accuracy_score(y_test, pred),
                "Precision": precision_score(
                    y_test, pred, average="macro", zero_division=0
                ),
                "Recall": recall_score(
                    y_test, pred, average="macro", zero_division=0
                ),
                "F1": f1_score(y_test, pred, average="macro", zero_division=0),
            }
        )

    results = pd.DataFrame(rows)
    return models, results, X_test, y_test, predictions


def risk_flower(label: str) -> tuple[str, str]:
    normalized = label.lower()
    if "low" in normalized:
        return "🌱", "Teal bud"
    if "moderate" in normalized or "medium" in normalized:
        return "🌼", "Yellow half-bloom"
    return "🌸", "Pink full bloom"


def html_card(title: str, body: str, ready: bool = True) -> str:
    state = "ready" if ready else "loading"

    title_classes = {
        "Risk Pattern": "risk-result-card",
        "AI Confidence": "confidence-result-card",
        "Top Biomarker": "top-biomarker-card",
        "Explainable AI": "explanation-result-card",
        "Mochi's Notebook": "notebook-card",
    }

    extra_class = title_classes.get(title, "")

    return f"""
    <div class="result-card {state} {extra_class}">
        <h3>{title}</h3>
        {body}
    </div>
    """


def optional_asset(path: Path, caption: str = ""):
    if path.exists():
        st.image(str(path), caption=caption or None, use_container_width=True)



BIOMARKER_LABELS = {
    "hippocampal_volume": "Hippocampal volume",
    "cortical_thickness": "Cortical thickness",
    "pet_biomarker": "PET biomarker",
}


def biomarker_summary_table(
    df: pd.DataFrame,
    selected_row: pd.DataFrame,
) -> pd.DataFrame:
    """Compare one sample's imaging biomarkers with the current dataset."""
    rows = []

    for feature, label in BIOMARKER_LABELS.items():
        value = float(selected_row.iloc[0][feature])
        series = pd.to_numeric(df[feature], errors="coerce").dropna()
        median = float(series.median())

        # Percentile = percentage of dataset values at or below this value.
        percentile = float((series <= value).mean() * 100)

        rows.append(
            {
                "Biomarker": label,
                "Selected value": value,
                "Dataset median": median,
                "Dataset percentile": percentile,
            }
        )

    return pd.DataFrame(rows)


def make_biomarker_figure(summary: pd.DataFrame):
    """Create a normalized biomarker view for features with different units."""
    plot_data = summary.sort_values("Dataset percentile")

    fig, ax = plt.subplots(figsize=(8.4, 3.8))
    bars = ax.barh(
        plot_data["Biomarker"],
        plot_data["Dataset percentile"],
    )

    ax.axvline(
        50,
        linestyle="--",
        linewidth=1.2,
        alpha=0.7,
        label="Dataset median percentile",
    )
    ax.set_xlim(0, 100)
    ax.set_xlabel("Percentile within this dataset")
    ax.set_title("Selected sample compared with the current dataset")
    ax.grid(axis="x", alpha=0.22)

    for bar, (_, row) in zip(bars, plot_data.iterrows()):
        percentile = float(row["Dataset percentile"])
        selected_value = float(row["Selected value"])
        label_x = min(percentile + 2, 94)

        ax.text(
            label_x,
            bar.get_y() + bar.get_height() / 2,
            f"{percentile:.0f}th · {selected_value:,.3g}",
            va="center",
            fontsize=9,
        )

    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def render_biomarker_card(
    df: pd.DataFrame,
    selected_row: pd.DataFrame,
    participant_id: str,
):
    """Render the imaging biomarker visualization as a dashboard card."""
    summary = biomarker_summary_table(df, selected_row)

    with st.container(
        border=True,
        key="biomarker_visualization_card",
    ):
        st.markdown("### 🧠 Biomarker visualization")
        st.caption(
            f"Imaging biomarkers for sample {participant_id}. "
            "Percentiles place measurements with different units on one scale."
        )

        fig = make_biomarker_figure(summary)
        st.pyplot(fig, clear_figure=True, use_container_width=True)

        display_table = summary.copy()
        display_table["Selected value"] = display_table["Selected value"].map(
            lambda value: f"{value:,.3f}".rstrip("0").rstrip(".")
        )
        display_table["Dataset median"] = display_table["Dataset median"].map(
            lambda value: f"{value:,.3f}".rstrip("0").rstrip(".")
        )
        display_table["Dataset percentile"] = display_table[
            "Dataset percentile"
        ].map(lambda value: f"{value:.0f}th")

        st.dataframe(
            display_table,
            use_container_width=True,
            hide_index=True,
        )

        st.caption(
            "A higher percentile only means a higher numerical value in this "
            "dataset. It does not automatically mean healthier or riskier."
        )

def render_fireflies():
    positions = [
        (8, 22, 0.0),
        (14, 65, 1.2),
        (21, 43, 2.1),
        (29, 76, 0.8),
        (38, 30, 1.6),
        (46, 57, 2.7),
        (56, 18, 1.0),
        (64, 68, 2.2),
        (72, 39, 0.4),
        (82, 74, 1.9),
        (90, 25, 2.9),
    ]
    dots = "".join(
        f'<span class="firefly" style="left:{x}%;top:{y}%;animation-delay:{delay}s"></span>'
        for x, y, delay in positions
    )
    st.markdown(
        f"""
        <div class="firefly-field">
            {dots}
            <div class="flower-row">🌷 🌼 🌸 🌻 🌺</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ------------------------------------------------------------
# SESSION STATE
# ------------------------------------------------------------
if "analysis_ready" not in st.session_state:
    st.session_state.analysis_ready = False

if "show_review" not in st.session_state:
    st.session_state.show_review = False


# ------------------------------------------------------------
# HEADER
# ------------------------------------------------------------
# The large title and safety bar are shown before analysis.
# After results are ready, they disappear to give the dashboard
# more room. The safety note remains available in the sidebar.
if not st.session_state.analysis_ready:
    st.markdown(
        """
        <div class="title-card">
            <h1>NeuroMed Scan AI</h1>
            <p>
                Explainable multimodal brain-imaging model for early
                cognitive-risk pattern detection.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="safety-note">
            <strong>Research and education prototype only.</strong>
            This dashboard does not diagnose dementia, provide medical advice,
            or recommend changing medication.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------
with st.sidebar:
    st.header("Dashboard controls")

    mode = st.radio(
        "Presentation mode",
        ["Story Mode", "Science Mode"],
        index=0,
    )

    source = st.radio(
        "Data source",
        ["Demo dataset", "Upload CSV"],
        index=0,
    )

    uploaded = None
    if source == "Upload CSV":
        uploaded = st.file_uploader(
            "Upload your cleaned CSV",
            type=["csv"],
            key="upload_data_widget",
        )

    st.caption(
        "Expected columns: age, cognitive_score, medication_burden, "
        "hippocampal_volume, cortical_thickness, pet_biomarker, risk_label."
    )

    st.divider()
    st.subheader("Story controls")
    reveal_speed = st.slider(
        "Card reveal speed",
        min_value=0.1,
        max_value=1.0,
        value=0.35,
        step=0.05,
    )

    st.divider()

    with st.expander("ⓘ About & safety", expanded=False):
        st.caption(
            "Research and education prototype only. This dashboard does not "
            "diagnose dementia, provide medical advice, or recommend changing "
            "medication."
        )

    missing_approved_assets = [
        label
        for label, path in {
            "Upload button": APPROVED_ASSETS["upload_button"],
            "Analysis button": APPROVED_ASSETS["analysis_button"],
            "Results button": APPROVED_ASSETS["results_button"],
            "Stars": APPROVED_ASSETS["stars"],
            "Twilight meadow": APPROVED_ASSETS["twilight_meadow"],
            "Night meadow": APPROVED_ASSETS["night_meadow"],
            "Midnight meadow": APPROVED_ASSETS["midnight_meadow"],
        }.items()
        if path is None
    ]

    if missing_approved_assets:
        with st.expander("Asset check", expanded=False):
            st.caption(
                "Not found: " + ", ".join(missing_approved_assets)
            )

    if st.session_state.analysis_ready:
        st.success("Results ready")

        if st.button(
            "Run a new analysis",
            use_container_width=True,
            key="new_analysis_button",
        ):
            st.session_state.analysis_ready = False
            st.session_state.show_review = False
            st.rerun()


# ------------------------------------------------------------
# APPROVED VISUAL ASSETS
# ------------------------------------------------------------
if not st.session_state.analysis_ready:
    active_meadow = APPROVED_ASSETS["twilight_meadow"]
elif mode == "Story Mode":
    active_meadow = (
        APPROVED_ASSETS["night_meadow"]
        or APPROVED_ASSETS["twilight_meadow"]
    )
else:
    active_meadow = (
        APPROVED_ASSETS["midnight_meadow"]
        or APPROVED_ASSETS["night_meadow"]
        or APPROVED_ASSETS["twilight_meadow"]
    )

meadow_uri = asset_data_uri(
    str(active_meadow) if active_meadow else None
)
stars_uri = asset_data_uri(
    str(APPROVED_ASSETS["stars"])
    if APPROVED_ASSETS["stars"]
    else None
)
comet_uri = asset_data_uri(
    str(APPROVED_ASSETS["comet"])
    if APPROVED_ASSETS["comet"]
    else None
)
upload_button_uri = asset_data_uri(
    str(APPROVED_ASSETS["upload_button"])
    if APPROVED_ASSETS["upload_button"]
    else None
)
analysis_button_uri = asset_data_uri(
    str(APPROVED_ASSETS["analysis_button"])
    if APPROVED_ASSETS["analysis_button"]
    else None
)
results_button_uri = asset_data_uri(
    str(APPROVED_ASSETS["results_button"])
    if APPROVED_ASSETS["results_button"]
    else None
)

background_css = ""

if meadow_uri:
    background_css += f"""
    [data-testid="stAppViewContainer"] {{
        background-image:
            linear-gradient(
                rgba(44, 39, 64, .08),
                rgba(36, 43, 62, .17)
            ),
            url('{meadow_uri}') !important;
        background-size: cover !important;
        background-position: center center !important;
        background-repeat: no-repeat !important;
        background-attachment: fixed !important;
    }}

    [data-testid="stAppViewContainer"] > .main {{
        background: transparent !important;
    }}
    """

if stars_uri:
    background_css += f"""
    .stApp::before {{
        content: "";
        position: fixed;
        inset: 0;
        z-index: 0;
        pointer-events: none;
        background-image: url('{stars_uri}');
        background-repeat: no-repeat;
        background-position: center top;
        background-size: cover;
        opacity: .30;
        animation: gentle-star-glow 4.8s ease-in-out infinite alternate;
    }}

    @keyframes gentle-star-glow {{
        from {{ opacity: .20; }}
        to {{ opacity: .42; }}
    }}
    """

if comet_uri:
    background_css += f"""
    .stApp::after {{
        content: "";
        position: fixed;
        top: 4vh;
        right: 5vw;
        width: min(25vw, 340px);
        height: min(18vw, 220px);
        z-index: 0;
        pointer-events: none;
        background-image: url('{comet_uri}');
        background-repeat: no-repeat;
        background-position: center;
        background-size: contain;
        opacity: .42;
        animation: comet-drift 12s ease-in-out infinite alternate;
    }}

    @keyframes comet-drift {{
        from {{ transform: translate(0, 0) rotate(-2deg); }}
        to {{ transform: translate(-28px, 15px) rotate(1deg); }}
    }}
    """

button_css = ""

if upload_button_uri:
    button_css += f"""
    .st-key-upload_data_widget [data-testid="stFileUploaderDropzone"] {{
        min-height: 92px !important;
        border: 0 !important;
        border-radius: 22px !important;
        background:
            url('{upload_button_uri}')
            center / contain no-repeat !important;
        box-shadow: none !important;
        cursor: pointer !important;
    }}

    .st-key-upload_data_widget
    [data-testid="stFileUploaderDropzoneInstructions"],
    .st-key-upload_data_widget
    [data-testid="stBaseButton-secondary"] {{
        opacity: 0 !important;
    }}
    """

if analysis_button_uri:
    button_css += f"""
    .st-key-begin_analysis_button button {{
        min-height: 96px !important;
        border: 0 !important;
        background:
            url('{analysis_button_uri}')
            center / contain no-repeat !important;
        color: transparent !important;
        box-shadow: none !important;
        transform: none !important;
    }}

    .st-key-begin_analysis_button button:hover {{
        filter: brightness(1.07)
            drop-shadow(0 8px 12px rgba(79, 51, 46, .20)) !important;
        transform: translateY(-2px) !important;
    }}

    .st-key-begin_analysis_button button:active {{
        transform: translateY(2px) !important;
    }}
    """

if results_button_uri:
    button_css += f"""
    .st-key-results_image_button button {{
        min-height: 90px !important;
        border: 0 !important;
        background:
            url('{results_button_uri}')
            center / contain no-repeat !important;
        color: transparent !important;
        box-shadow: none !important;
        transform: none !important;
    }}

    .st-key-results_image_button button:hover {{
        filter: brightness(1.07)
            drop-shadow(0 8px 12px rgba(79, 51, 46, .20)) !important;
        transform: translateY(-2px) !important;
    }}

    .st-key-results_image_button button:active {{
        transform: translateY(2px) !important;
    }}
    """

if background_css or button_css:
    st.markdown(
        f"""
        <style>
        .stApp {{
            position: relative;
        }}

        [data-testid="stAppViewContainer"] > .main,
        [data-testid="stSidebar"] {{
            position: relative;
            z-index: 1;
        }}

        {background_css}
        {button_css}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ------------------------------------------------------------
# MODE-SPECIFIC FONTS
# ------------------------------------------------------------
if mode == "Story Mode":
    st.markdown(
        """
        <style>
        .stApp,
        .stApp p,
        .stApp label,
        .stApp input,
        .stApp textarea,
        .stApp select,
        .stApp [data-testid="stDataFrame"],
        .stApp [data-testid="stMetricValue"],
        .stApp [data-testid="stMetricLabel"] {
            font-family: 'Quicksand', sans-serif !important;
        }

        .stApp h1,
        .stApp h2,
        .stApp h3,
        .stApp div.stButton > button,
        .title-card h1,
        .story-strip strong {
            font-family: 'Balsamiq Sans', cursive !important;
        }

        .result-card {
            border-color: rgba(123, 86, 62, .68) !important;
            box-shadow:
                0 12px 28px rgba(58, 43, 76, .14),
                inset 0 1px 0 rgba(255,255,255,.74) !important;
        }

        .risk-result-card {
            background:
                linear-gradient(
                    145deg,
                    rgba(204, 239, 232, .94),
                    rgba(232, 248, 244, .88)
                ) !important;
        }

        .confidence-result-card {
            background:
                linear-gradient(
                    145deg,
                    rgba(229, 218, 255, .94),
                    rgba(244, 239, 255, .88)
                ) !important;
        }

        .top-biomarker-card {
            background:
                linear-gradient(
                    145deg,
                    rgba(255, 236, 177, .94),
                    rgba(255, 248, 222, .88)
                ) !important;
        }

        .explanation-result-card {
            background:
                linear-gradient(
                    145deg,
                    rgba(249, 218, 231, .94),
                    rgba(255, 241, 247, .89)
                ) !important;
        }

        .notebook-card {
            font-family: 'Balsamiq Sans', cursive !important;
            background:
                repeating-linear-gradient(
                    to bottom,
                    rgba(255, 249, 238, .97) 0px,
                    rgba(255, 249, 238, .97) 29px,
                    rgba(183, 208, 221, .38) 30px
                ) !important;
            border-left: 5px solid rgba(238, 166, 180, .82) !important;
        }

        .notebook-card,
        .notebook-card * {
            font-family: 'Balsamiq Sans', cursive !important;
        }

        .st-key-biomarker_visualization_card {
            background:
                linear-gradient(
                    145deg,
                    rgba(219, 240, 255, .94),
                    rgba(241, 247, 255, .90)
                ) !important;
            border: 1.5px solid rgba(123, 86, 62, .68) !important;
            border-radius: 22px !important;
            padding: 1rem 1.1rem !important;
            box-shadow:
                0 12px 28px rgba(58, 43, 76, .14),
                inset 0 1px 0 rgba(255,255,255,.74) !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        """
        <style>
        .stApp,
        .stApp p,
        .stApp label,
        .stApp input,
        .stApp textarea,
        .stApp select,
        .stApp h1,
        .stApp h2,
        .stApp h3,
        .stApp div.stButton > button,
        .stApp [data-testid="stDataFrame"],
        .stApp [data-testid="stMetricValue"],
        .stApp [data-testid="stMetricLabel"] {
            font-family: 'Montserrat', sans-serif !important;
        }

        .result-card,
        .risk-result-card,
        .confidence-result-card,
        .top-biomarker-card,
        .explanation-result-card {
            background: rgba(255, 255, 255, .92) !important;
            border-color: rgba(111, 101, 119, .40) !important;
            box-shadow:
                0 10px 24px rgba(49, 45, 58, .10),
                inset 0 1px 0 rgba(255,255,255,.90) !important;
        }

        .notebook-card {
            font-family: 'Balsamiq Sans', cursive !important;
            background:
                repeating-linear-gradient(
                    to bottom,
                    rgba(255, 254, 248, .97) 0px,
                    rgba(255, 254, 248, .97) 29px,
                    rgba(183, 202, 214, .30) 30px
                ) !important;
            border-left: 5px solid rgba(190, 159, 176, .70) !important;
        }

        .notebook-card,
        .notebook-card * {
            font-family: 'Balsamiq Sans', cursive !important;
        }

        .st-key-biomarker_visualization_card {
            background: rgba(255, 255, 255, .94) !important;
            border: 1.5px solid rgba(111, 101, 119, .40) !important;
            border-radius: 20px !important;
            padding: 1rem 1.1rem !important;
            box-shadow:
                0 10px 24px rgba(49, 45, 58, .10),
                inset 0 1px 0 rgba(255,255,255,.90) !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ------------------------------------------------------------
# READY-STATE LAYOUT
# ------------------------------------------------------------
if st.session_state.analysis_ready:
    st.markdown(
        """
        <style>
        .main .block-container {
            padding-top: .25rem !important;
            padding-bottom: .3rem !important;
        }

        [data-testid="stTabs"] {
            height: calc(100vh - 150px) !important;
        }

        [data-testid="stTabs"] [data-baseweb="tab-panel"] {
            height: calc(100vh - 205px) !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------
if source == "Demo dataset":
    raw_df = make_demo_data()
    data_label = "Synthetic demo data"
else:
    if uploaded is None:
        st.info("Upload a CSV to begin, or switch to the demo dataset.")
        st.stop()
    raw_df = pd.read_csv(uploaded)
    data_label = uploaded.name

try:
    df, matched_columns = prepare_dataset(raw_df)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

st.caption(f"Using: **{data_label}** · {len(df)} complete rows")


# ------------------------------------------------------------
# BEGIN ANALYSIS
# ------------------------------------------------------------
begin = False

if not st.session_state.analysis_ready:
    left, center, right = st.columns([1, 1.2, 1])

    with center:
        begin = st.button(
            "Begin Analysis",
            use_container_width=True,
            key="begin_analysis_button",
        )
        st.markdown(
            '<div class="wood-button-hint">'
            'Press to train and compare the three models.'
            '</div>',
            unsafe_allow_html=True,
        )

if begin:
    st.session_state.analysis_ready = False
    st.session_state.show_review = False

    loading_box = st.empty()
    progress = st.progress(0, text="Preparing the dataset...")

    steps = [
        ("Aligning the telescope with the data constellation...", 16),
        ("Scanning the brain constellation...", 34),
        ("A meteor shower passes overhead...", 52),
        ("Comparing medication-only and imaging-only models...", 72),
        ("Training the combined explainable model...", 88),
        ("Preparing the result cards...", 100),
    ]

    for message, value in steps:
        loading_box.markdown(
            f"""
            <div class="story-strip">
                <strong>{message}</strong><br>
                <span class="small-muted">
                    The cards remain softly blurred until the results are ready.
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        progress.progress(value, text=message)
        time.sleep(reveal_speed)

    progress.empty()
    loading_box.empty()
    st.session_state.analysis_ready = True

    # Rerun immediately so the title/safety bar and Begin button
    # disappear as soon as the results are ready.
    st.rerun()

if not st.session_state.analysis_ready:
    placeholder_cols = st.columns(4)
    for col, title in zip(
        placeholder_cols,
        [
            "Risk Pattern",
            "AI Confidence",
            "Top Biomarker",
            "Biomarker View",
        ],
    ):
        with col:
            st.markdown(
                html_card(
                    title,
                    '<div class="big-number">Loading…</div>'
                    '<p class="small-muted">Waiting for analysis.</p>',
                    ready=False,
                ),
                unsafe_allow_html=True,
            )
    st.stop()


# ------------------------------------------------------------
# TRAIN MODELS
# ------------------------------------------------------------
with st.spinner("Training models..."):
    models, metrics_df, X_test, y_test, predictions = train_models(df)

combined_model, combined_features = models["Combined"]
combined_pred = predictions["Combined"]

combined_accuracy = float(
    metrics_df.loc[metrics_df["Model"] == "Combined", "Accuracy"].iloc[0]
)

participant_options = df["participant_id"].tolist()
selected_id = st.selectbox(
    "Choose a participant/sample to explain",
    participant_options,
    index=0,
)
selected_row = df.loc[df["participant_id"] == selected_id].iloc[[0]]

predicted_label = combined_model.predict(selected_row[combined_features])[0]
proba = combined_model.predict_proba(selected_row[combined_features])[0]
confidence = float(np.max(proba))
flower, flower_name = risk_flower(predicted_label)

importance_df = pd.DataFrame(
    {
        "Feature": combined_features,
        "Importance": combined_model.feature_importances_,
    }
).sort_values("Importance", ascending=False)

top_feature = str(importance_df.iloc[0]["Feature"])
top_importance = float(importance_df.iloc[0]["Importance"])


# ------------------------------------------------------------

# ------------------------------------------------------------
# SINGLE-SCREEN STORY / SCIENCE MODES
# ------------------------------------------------------------
if mode == "Story Mode":
    story_tabs = st.tabs(
        ["🌸 Results", "🧠 Biomarkers", "📊 Comparison", "✨ Ending"]
    )

    with story_tabs[0]:
        st.markdown(
            """
            <div class="story-strip">
                <strong>Results are ready.</strong>
                The result cards clear as the constellation analysis completes.
            </div>
            """,
            unsafe_allow_html=True,
        )

        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown(
                html_card(
                    "Risk Pattern",
                    f"""
                    <div class="risk-flower">{flower}</div>
                    <div class="big-number">{predicted_label}</div>
                    <p class="small-muted">
                        {flower_name} represents this research risk pattern.
                    </p>
                    """,
                ),
                unsafe_allow_html=True,
            )

        with c2:
            st.markdown(
                html_card(
                    "AI Confidence",
                    f"""
                    <div class="big-number">{confidence:.0%}</div>
                    <p class="small-muted">
                        Model certainty for this sample—not medical certainty.
                    </p>
                    """,
                ),
                unsafe_allow_html=True,
            )

        with c3:
            st.markdown(
                html_card(
                    "Top Biomarker",
                    f"""
                    <div class="big-number">
                        {top_feature.replace("_", " ").title()}
                    </div>
                    <p class="small-muted">
                        Largest overall importance in the combined model.
                    </p>
                    """,
                ),
                unsafe_allow_html=True,
            )

        st.markdown(
            html_card(
                "Explainable AI",
                f"""
                <p>
                    The model predicted <strong>{predicted_label}</strong> risk.
                    Its strongest overall feature was
                    <strong>{top_feature.replace("_", " ")}</strong>.
                </p>
                <p class="small-muted">
                    This explains the model's reasoning, not a diagnosis.
                </p>
                """,
            ),
            unsafe_allow_html=True,
        )

    with story_tabs[1]:
        summary = biomarker_summary_table(df, selected_row)

        left_bio, right_bio = st.columns([1.65, 1])

        with left_bio:
            fig = make_biomarker_figure(summary)
            fig.set_size_inches(7.2, 3.0)
            st.pyplot(fig, clear_figure=True, use_container_width=True)

        with right_bio:
            st.markdown(
                html_card(
                    "Biomarker View",
                    f"""
                    <p><strong>Sample:</strong> {selected_id}</p>
                    <p>
                        The chart compares hippocampal volume, cortical
                        thickness, and the PET biomarker by dataset percentile.
                    </p>
                    <p class="small-muted">
                        Higher does not automatically mean healthier or riskier.
                    </p>
                    """,
                ),
                unsafe_allow_html=True,
            )

            display_summary = summary.copy()
            display_summary["Selected value"] = display_summary[
                "Selected value"
            ].map(lambda value: f"{value:,.3g}")
            display_summary["Dataset percentile"] = display_summary[
                "Dataset percentile"
            ].map(lambda value: f"{value:.0f}th")

            st.dataframe(
                display_summary[
                    ["Biomarker", "Selected value", "Dataset percentile"]
                ],
                use_container_width=True,
                hide_index=True,
                height=165,
            )

    with story_tabs[2]:
        chart_col, note_col = st.columns([1.7, 1])

        with chart_col:
            fig, ax = plt.subplots(figsize=(7.0, 3.0))
            ax.bar(metrics_df["Model"], metrics_df["Accuracy"] * 100)
            ax.set_ylabel("Accuracy (%)")
            ax.set_ylim(0, 100)
            ax.set_title("Medication-only vs imaging-only vs combined")
            ax.grid(axis="y", alpha=0.25)

            for idx, value in enumerate(metrics_df["Accuracy"] * 100):
                ax.text(idx, value + 2, f"{value:.0f}%", ha="center", fontsize=9)

            fig.tight_layout()
            st.pyplot(fig, clear_figure=True, use_container_width=True)

        with note_col:
            best_model = metrics_df.sort_values(
                "Accuracy",
                ascending=False,
            ).iloc[0]["Model"]

            st.markdown(
                html_card(
                    "Model Comparison",
                    f"""
                    <div class="big-number">{best_model}</div>
                    <p>
                        This model produced the highest accuracy in this
                        particular test split.
                    </p>
                    <p class="small-muted">
                        Results may change with a different split or dataset.
                    </p>
                    """,
                ),
                unsafe_allow_html=True,
            )

            st.dataframe(
                metrics_df[
                    ["Model", "Accuracy", "Balanced Accuracy", "F1"]
                ].style.format(
                    {
                        "Accuracy": "{:.1%}",
                        "Balanced Accuracy": "{:.1%}",
                        "F1": "{:.1%}",
                    }
                ),
                use_container_width=True,
                hide_index=True,
                height=145,
            )

    with story_tabs[3]:
        ending_left, ending_right = st.columns([1.1, 1])

        with ending_left:
            st.markdown(
                """
                <div class="story-strip">
                    The result cards settle into the meadow as the final
                    data review becomes available.
                </div>
                """,
                unsafe_allow_html=True,
            )
            render_fireflies()

        with ending_right:
            if st.button(
                "Review Results",
                use_container_width=True,
                key="results_image_button",
            ):
                st.session_state.show_review = True

            if st.session_state.show_review:
                st.success(
                    "The cards fade into flowers and the data becomes fireflies."
                )
                st.dataframe(
                    selected_row[
                        [
                            "participant_id",
                            "age",
                            "cognitive_score",
                            "medication_burden",
                            "hippocampal_volume",
                            "cortical_thickness",
                            "pet_biomarker",
                        ]
                    ],
                    use_container_width=True,
                    hide_index=True,
                    height=185,
                )
            else:
                st.markdown(
                    html_card(
                        "Review Data",
                        """
                        <p>
                            Press <strong>Finish Story</strong> to reveal the
                            final data review.
                        </p>
                        <p class="small-muted">
                            The yellow fireflies will reform into the data.
                        </p>
                        """,
                    ),
                    unsafe_allow_html=True,
                )

else:
    science_tabs = st.tabs(
        [
            "Overview",
            "Biomarkers",
            "Models",
            "Explainability",
            "Data",
        ]
    )

    with science_tabs[0]:
        m1, m2, m3, m4 = st.columns(4)

        with m1:
            st.metric("Predicted risk", predicted_label)
        with m2:
            st.metric("AI confidence", f"{confidence:.1%}")
        with m3:
            st.metric("Combined accuracy", f"{combined_accuracy:.1%}")
        with m4:
            st.metric("Complete rows", len(df))

        st.markdown("#### Participant input")
        st.dataframe(
            selected_row[
                [
                    "participant_id",
                    "age",
                    "cognitive_score",
                    "medication_burden",
                    "hippocampal_volume",
                    "cortical_thickness",
                    "pet_biomarker",
                ]
            ],
            use_container_width=True,
            hide_index=True,
            height=180,
        )

    with science_tabs[1]:
        summary = biomarker_summary_table(df, selected_row)
        bio_left, bio_right = st.columns([1.65, 1])

        with bio_left:
            fig = make_biomarker_figure(summary)
            fig.set_size_inches(7.2, 3.0)
            st.pyplot(fig, clear_figure=True, use_container_width=True)

        with bio_right:
            st.dataframe(
                summary.style.format(
                    {
                        "Selected value": "{:,.3g}",
                        "Dataset median": "{:,.3g}",
                        "Dataset percentile": "{:.0f}",
                    }
                ),
                use_container_width=True,
                hide_index=True,
                height=220,
            )
            st.caption(
                "Percentiles compare numerical values within this dataset only."
            )

    with science_tabs[2]:
        left_models, right_models = st.columns([1.55, 1])

        with left_models:
            fig, ax = plt.subplots(figsize=(7.0, 3.1))
            x = np.arange(len(metrics_df))
            width = 0.16

            ax.bar(
                x - 2 * width,
                metrics_df["Accuracy"],
                width,
                label="Accuracy",
            )
            ax.bar(
                x - width,
                metrics_df["Balanced Accuracy"],
                width,
                label="Balanced Accuracy",
            )
            ax.bar(
                x,
                metrics_df["Precision"],
                width,
                label="Precision",
            )
            ax.bar(
                x + width,
                metrics_df["Recall"],
                width,
                label="Recall",
            )
            ax.bar(
                x + 2 * width,
                metrics_df["F1"],
                width,
                label="F1",
            )
            ax.set_xticks(x, metrics_df["Model"])
            ax.set_ylim(0, 1)
            ax.set_ylabel("Score")
            ax.set_title("Model performance")
            ax.legend(ncols=3, fontsize=8)
            ax.grid(axis="y", alpha=0.25)
            fig.tight_layout()

            st.pyplot(fig, clear_figure=True, use_container_width=True)

        with right_models:
            st.dataframe(
                metrics_df.style.format(
                    {
                        "Accuracy": "{:.1%}",
                        "Balanced Accuracy": "{:.1%}",
                        "Precision": "{:.1%}",
                        "Recall": "{:.1%}",
                        "F1": "{:.1%}",
                    }
                ),
                use_container_width=True,
                hide_index=True,
                height=180,
            )

            labels = sorted(y_test.unique().tolist())
            cm = confusion_matrix(y_test, combined_pred, labels=labels)

            fig_cm, ax_cm = plt.subplots(figsize=(3.5, 2.5))
            display = ConfusionMatrixDisplay(
                confusion_matrix=cm,
                display_labels=labels,
            )
            display.plot(ax=ax_cm, cmap="Blues", colorbar=False)
            ax_cm.set_title("Combined model")
            fig_cm.tight_layout()
            st.pyplot(fig_cm, clear_figure=True, use_container_width=True)

    with science_tabs[3]:
        explanation_left, explanation_right = st.columns([1.45, 1])

        with explanation_left:
            fig2, ax2 = plt.subplots(figsize=(6.8, 3.2))
            ordered = importance_df.sort_values("Importance")
            ax2.barh(
                ordered["Feature"].str.replace("_", " ").str.title(),
                ordered["Importance"],
            )
            ax2.set_xlabel("Feature importance")
            ax2.set_title("Which inputs mattered most?")
            ax2.grid(axis="x", alpha=0.25)
            fig2.tight_layout()
            st.pyplot(fig2, clear_figure=True, use_container_width=True)

        with explanation_right:
            st.markdown(
                html_card(
                    "Mochi's Notebook",
                    f"""
                    <p>
                        The model predicted <strong>{predicted_label}</strong>
                        risk with <strong>{confidence:.1%} confidence</strong>
                        for sample <strong>{selected_id}</strong>.
                    </p>
                    <p>
                        The strongest overall feature was
                        <strong>{top_feature.replace("_", " ")}</strong>.
                    </p>
                    <p class="small-muted">
                        Research explanation only—not a diagnosis.
                    </p>
                    """,
                ),
                unsafe_allow_html=True,
            )

    with science_tabs[4]:
        data_left, data_right = st.columns([2, 1])

        with data_left:
            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True,
                height=320,
            )

        with data_right:
            st.markdown("#### Matched columns")
            st.json(matched_columns)


