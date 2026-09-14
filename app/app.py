from __future__ import annotations

import sys
from pathlib import Path
from textwrap import dedent
import html
import duckdb
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DB_FILE

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "lightgbm_main_model.joblib"
)


# ============================================================
# HTML RENDERER
# ============================================================

def render_html(content, container=None):
    html = dedent(content).strip()

    if container is None:
        st.html(html)
    else:
        with container:
            st.html(html)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Payment Risk Intelligence",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# DESIGN SYSTEM
# ============================================================

render_html(
    """
    <style>

    /* ========================================================
       APP
    ======================================================== */

    .stApp {
        background:
            radial-gradient(
                circle at 15% 0%,
                rgba(53, 65, 95, 0.22),
                transparent 28%
            ),
            radial-gradient(
                circle at 85% 10%,
                rgba(177, 153, 99, 0.08),
                transparent 22%
            ),
            linear-gradient(
                135deg,
                #080c14 0%,
                #0b111d 45%,
                #0d1421 100%
            );

        color: #f2eee7;
    }

    .block-container {
        max-width: 1480px;
        padding-top: 2rem;
        padding-bottom: 4rem;
        padding-left: 2.5rem;
        padding-right: 2.5rem;
    }


    /* ========================================================
       SIDEBAR
    ======================================================== */

    section[data-testid="stSidebar"] {
        background:
            linear-gradient(
                180deg,
                #0d1421 0%,
                #090e17 100%
            );

        border-right:
            1px solid rgba(201, 179, 126, 0.16);
    }

    section[data-testid="stSidebar"] > div {
        padding-top: 1.8rem;
    }

    section[data-testid="stSidebar"] * {
        color: #e9e4da;
    }


    /* ========================================================
       SELECT BOXES
    ======================================================== */

 /* Dropdowns */

div[data-baseweb="select"] > div {
    background: #111827 !important;
    border: 1px solid rgba(201, 179, 126, 0.22) !important;
    border-radius: 10px !important;
    color: #f5f1e9 !important;
}

div[data-baseweb="select"] > div:hover {
    border-color: rgba(201, 179, 126, 0.50) !important;
}


/* Search transaction box */

section[data-testid="stSidebar"] div[data-testid="stTextInput"] div[data-baseweb="input"],
section[data-testid="stSidebar"] div[data-testid="stTextInput"] div[data-baseweb="base-input"] {
    background: #111827 !important;
    background-color: #111827 !important;

    border: 1px solid rgba(201, 179, 126, 0.22) !important;
    border-radius: 10px !important;

    box-shadow: none !important;
}

section[data-testid="stSidebar"] div[data-testid="stTextInput"] input {
    background: #111827 !important;
    background-color: #111827 !important;

    color: #f5f1e9 !important;
    -webkit-text-fill-color: #f5f1e9 !important;

    caret-color: #d8be7a !important;

    border: none !important;
    outline: none !important;
    box-shadow: none !important;
}

section[data-testid="stSidebar"] div[data-testid="stTextInput"] input::placeholder {
    color: #687386 !important;
    -webkit-text-fill-color: #687386 !important;
    opacity: 1 !important;
}

section[data-testid="stSidebar"] div[data-testid="stTextInput"] div[data-baseweb="input"]:focus-within,
section[data-testid="stSidebar"] div[data-testid="stTextInput"] div[data-baseweb="base-input"]:focus-within {
    border-color: #c9b37e !important;

    box-shadow:
        0 0 0 2px rgba(201, 179, 126, 0.10) !important;
}
    div[data-testid="stSelectbox"] span {
        color: #f5f1e9 !important;
    }

    div[data-testid="stSelectbox"] svg {
        fill: #aeb6c2 !important;
        color: #aeb6c2 !important;
    }


    /* ========================================================
       TEXT INPUT
    ======================================================== */

    div[data-testid="stTextInput"] div[data-baseweb="input"] {
        background: #111827 !important;

        border:
            1px solid rgba(201, 179, 126, 0.22) !important;

        border-radius: 10px !important;

        box-shadow: none !important;

        overflow: hidden !important;
    }

    div[data-testid="stTextInput"] div[data-baseweb="input"]:hover {
        border-color:
            rgba(201, 179, 126, 0.50) !important;
    }

    div[data-testid="stTextInput"] div[data-baseweb="input"]:focus-within {
        border-color: #c9b37e !important;

        box-shadow:
            0 0 0 2px rgba(201, 179, 126, 0.10) !important;
    }

    div[data-testid="stTextInput"] input {
        background: transparent !important;

        color: #f5f1e9 !important;

        -webkit-text-fill-color:
            #f5f1e9 !important;

        caret-color:
            #d8be7a !important;

        border: none !important;

        box-shadow: none !important;

        outline: none !important;
    }

    div[data-testid="stTextInput"] input::placeholder {
        color: #596577 !important;
        opacity: 1 !important;
    }

    div[data-testid="stTextInput"] input::selection {
        background: #c9b37e !important;
        color: #080c14 !important;
    }

    div[data-testid="stTextInput"] input:-webkit-autofill,
    div[data-testid="stTextInput"] input:-webkit-autofill:hover,
    div[data-testid="stTextInput"] input:-webkit-autofill:focus {
        -webkit-text-fill-color:
            #f5f1e9 !important;

        -webkit-box-shadow:
            0 0 0 1000px #111827 inset !important;

        caret-color:
            #d8be7a !important;
    }


    /* ========================================================
       LABELS
    ======================================================== */

    label,
    .stCaption {
        color: #a8afba !important;
    }


    /* ========================================================
       HEADINGS
    ======================================================== */

    h1, h2, h3 {
        color: #f4efe5;
        letter-spacing: -0.02em;
    }


    /* ========================================================
       DIVIDERS
    ======================================================== */

    hr {
        border: 0;

        border-top:
            1px solid rgba(255, 255, 255, 0.08);
    }


    /* ========================================================
       TABS
    ======================================================== */

    button[data-baseweb="tab"] {
        color: #8f98a7;
        font-weight: 500;
        padding-left: 0;
        padding-right: 2rem;
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        color: #e7d7ad;
    }

    div[data-baseweb="tab-highlight"] {
        background-color: #c9b37e;
    }

    div[data-baseweb="tab-border"] {
        background-color:
            rgba(255, 255, 255, 0.08);
    }


    /* ========================================================
       TABLES
    ======================================================== */

    div[data-testid="stDataFrame"] {
        border:
            1px solid rgba(255, 255, 255, 0.07);

        border-radius: 12px;

        overflow: hidden;
    }


    /* ========================================================
       DARK KV TABLE (CUSTOM)
    ======================================================== */

    .dark-kv-table {
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
        overflow: hidden;
        border-radius: 14px;
        border: 1px solid rgba(201, 179, 126, 0.12);
        background: rgba(10, 17, 29, 0.88);
    }

    .dark-kv-table thead th {
        background: rgba(255, 255, 255, 0.04);
        color: #8f99a7;
        text-align: left;
        font-size: 0.82rem;
        font-weight: 500;
        padding: 14px 16px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
    }

    .dark-kv-table tbody td {
        background: rgba(9, 14, 23, 0.95);
        color: #ece7de;
        padding: 14px 16px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.06);
    }

    .dark-kv-table tbody tr td:first-child {
        color: #b7c0cd;
        width: 42%;
    }

    .dark-kv-table tbody tr:last-child td {
        border-bottom: none;
    }

    .dark-kv-table th + th,
    .dark-kv-table td + td {
        border-left: 1px solid rgba(255, 255, 255, 0.05);
    }


    /* ========================================================
       DARK TABLE (GENERIC)
    ======================================================== */

    .table-container {
        width: 100%;
        overflow-x: auto;
        border-radius: 14px;
        border: 1px solid rgba(201, 179, 126, 0.12);
        background: rgba(10, 17, 29, 0.88);
        margin-bottom: 1rem;
    }

    .dark-table {
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
    }

    .dark-table thead th {
        background: rgba(255, 255, 255, 0.04);
        color: #8f99a7;
        text-align: left;
        font-size: 0.82rem;
        font-weight: 500;
        padding: 14px 16px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        white-space: nowrap;
    }

    .dark-table tbody td {
        background: transparent;
        color: #ece7de;
        padding: 14px 16px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.06);
        font-size: 0.88rem;
    }

    .dark-table tbody tr:last-child td {
        border-bottom: none;
    }

    .dark-table th + th,
    .dark-table td + td {
        border-left: 1px solid rgba(255, 255, 255, 0.05);
    }


    /* ========================================================
       PREMIUM COMPONENTS
    ======================================================== */

    .eyebrow {
        color: #c9b37e;

        font-size: 0.70rem;

        letter-spacing: 0.18em;

        text-transform: uppercase;

        font-weight: 600;

        margin-bottom: 0.65rem;
    }


    .hero-title {
        font-size: 2.7rem;

        line-height: 1.05;

        font-weight: 650;

        color: #f4efe5;

        letter-spacing: -0.035em;

        margin: 0;
    }


    .hero-subtitle {
        margin-top: 0.8rem;

        color: #9ca5b2;

        max-width: 760px;

        font-size: 0.98rem;

        line-height: 1.6;
    }


    /* ========================================================
       TRANSACTION STRIP
    ======================================================== */

    .transaction-strip {
        display: flex;

        align-items: center;

        gap: 0.65rem;

        width: 100%;

        background:
            linear-gradient(
                90deg,
                rgba(201, 179, 126, 0.055),
                rgba(255, 255, 255, 0.018)
            );

        border:
            1px solid rgba(201, 179, 126, 0.13);

        border-radius: 12px;

        padding:
            0.85rem 1.1rem;

        margin-top: 1.4rem;

        margin-bottom: 1.5rem;

        color: #98a2b3;

        font-size: 0.82rem;

        box-shadow:
            0 8px 25px rgba(0, 0, 0, 0.12);
    }


    .transaction-meta {
        color: #8691a1;
    }


    .transaction-divider {
        color: #485363;
        font-size: 0.9rem;
    }


    .transaction-status {
        margin-left: auto;

        display: flex;

        align-items: center;

        gap: 0.35rem;
    }


    .gold-text {
        color: #dfc98f;

        font-weight: 600;
    }


    /* ========================================================
       METRIC CARDS
    ======================================================== */

    .metric-card {
        background:
            linear-gradient(
                145deg,
                rgba(255, 255, 255, 0.055),
                rgba(255, 255, 255, 0.020)
            );

        border:
            1px solid rgba(255, 255, 255, 0.075);

        border-radius: 16px;

        padding: 1.25rem 1.35rem;

        min-height: 124px;

        box-shadow:
            0 10px 30px rgba(0, 0, 0, 0.18);

        transition:
            border-color 0.20s ease,
            transform 0.20s ease;
    }


    .metric-card:hover {
        border-color:
            rgba(201, 179, 126, 0.28);

        transform:
            translateY(-1px);
    }


    .metric-label {
        color: #8f99a7;

        font-size: 0.73rem;

        text-transform: uppercase;

        letter-spacing: 0.10em;

        font-weight: 600;
    }


    .metric-value {
        color: #f7f2e9;

        font-size: 1.85rem;

        line-height: 1.15;

        margin-top: 0.55rem;

        font-weight: 520;

        letter-spacing: -0.025em;
    }


    .metric-note {
        color: #6f7a89;

        font-size: 0.74rem;

        margin-top: 0.6rem;
    }


    .critical {
        color: #e0c27d;
    }


    /* ========================================================
       SECTION + DETAIL CARDS
    ======================================================== */

    .section-title {
        font-size: 1.2rem;

        color: #eee9df;

        font-weight: 600;

        margin-bottom: 1rem;
    }


    .detail-card {
        background:
            rgba(255, 255, 255, 0.025);

        border:
            1px solid rgba(255, 255, 255, 0.065);

        border-radius: 14px;

        padding: 1.1rem 1.2rem;

        min-height: 90px;
    }


    .detail-label {
        color: #778292;

        font-size: 0.71rem;

        letter-spacing: 0.08em;

        text-transform: uppercase;

        font-weight: 600;
    }


    .detail-value {
        color: #ece7de;

        font-size: 1.22rem;

        margin-top: 0.35rem;

        font-weight: 500;
    }


    /* ========================================================
       STATUS PILLS
    ======================================================== */

    .status-pill {
        display: inline-flex;

        align-items: center;

        padding:
            0.34rem 0.66rem;

        border-radius: 999px;

        font-size: 0.70rem;

        letter-spacing: 0.03em;

        white-space: nowrap;
    }


    .status-critical {
        background:
            rgba(201, 179, 126, 0.11);

        border:
            1px solid rgba(201, 179, 126, 0.30);

        color: #dfc98f;
    }


    .status-neutral {
        background:
            rgba(148, 163, 184, 0.08);

        border:
            1px solid rgba(148, 163, 184, 0.18);

        color: #b7c0cd;
    }


    /* ========================================================
       INSIGHT
    ======================================================== */

    .insight-box {
        background:
            linear-gradient(
                135deg,
                rgba(201, 179, 126, 0.07),
                rgba(255, 255, 255, 0.025)
            );

        border-left:
            3px solid #bca66f;

        border-radius: 10px;

        padding: 1rem 1.15rem;

        color: #c7ced7;

        margin-bottom: 1.2rem;
    }


    /* ========================================================
       FOOTER
    ======================================================== */

    .footer-note {
        color: #687383;

        font-size: 0.76rem;

        margin-top: 2rem;

        padding-top: 1.2rem;

        border-top:
            1px solid rgba(255, 255, 255, 0.06);
    }


    /* ========================================================
       STREAMLIT CLEANUP
    ======================================================== */

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    header[data-testid="stHeader"] {
        background: transparent;
    }

    </style>
    """
)


# ============================================================
# MODEL FEATURES
# ============================================================

NUMERIC_FEATURES = [
    "amount",
    "log_amount",
    "distance_from_home",
    "transaction_hour",
    "weekend_transaction",
    "card_present",
    "customer_prior_transactions",
    "minutes_since_previous_transaction",
    "previous_customer_amount",
    "customer_avg_amount_prior",
    "customer_std_amount_prior",
    "amount_vs_customer_average",
    "amount_zscore_vs_customer_history",
    "calculated_txn_count_1h",
    "calculated_amount_1h",
    "calculated_txn_count_24h",
    "calculated_amount_24h",
    "new_merchant_for_customer_flag",
    "overnight_transaction_flag",
]

CATEGORICAL_FEATURES = [
    "merchant_category",
    "merchant_type",
    "merchant",
    "currency",
    "country",
    "city_size",
    "card_type",
    "device",
    "channel",
]

FEATURES = (
    NUMERIC_FEATURES
    + CATEGORICAL_FEATURES
)


FRIENDLY_NAMES = {
    "amount": "Transaction amount",
    "log_amount": "Transaction amount scale",
    "distance_from_home": "Distance from home",
    "transaction_hour": "Transaction hour",
    "weekend_transaction": "Weekend transaction",
    "card_present": "Card present",
    "customer_prior_transactions": "Prior customer activity",
    "minutes_since_previous_transaction": "Time since previous transaction",
    "previous_customer_amount": "Previous transaction amount",
    "customer_avg_amount_prior": "Historical average amount",
    "customer_std_amount_prior": "Historical spend variation",
    "amount_vs_customer_average": "Amount vs customer average",
    "amount_zscore_vs_customer_history": "Amount anomaly score",
    "calculated_txn_count_1h": "Prior-hour transactions",
    "calculated_amount_1h": "Prior-hour spend",
    "calculated_txn_count_24h": "Prior 24h transactions",
    "calculated_amount_24h": "Prior 24h spend",
    "new_merchant_for_customer_flag": "New merchant",
    "overnight_transaction_flag": "Overnight activity",
    "merchant_category": "Merchant category",
    "merchant_type": "Merchant type",
    "merchant": "Merchant",
    "currency": "Currency",
    "country": "Country",
    "city_size": "City size",
    "card_type": "Card type",
    "device": "Device type",
    "channel": "Payment channel",
}


# ============================================================
# MODEL + DATABASE
# ============================================================

@st.cache_resource
def load_model():

    return joblib.load(
        MODEL_PATH
    )


model = load_model()


def query_database(
    query: str,
    params=None,
):

    connection = duckdb.connect(
        str(DB_FILE),
        read_only=True,
    )

    try:

        if params is None:

            return connection.execute(
                query
            ).fetchdf()

        return connection.execute(
            query,
            params,
        ).fetchdf()

    finally:

        connection.close()


def prepare_categories(
    dataframe,
):

    dataframe = dataframe.copy()

    for index, column in enumerate(
        CATEGORICAL_FEATURES
    ):

        dataframe[column] = (
            dataframe[column]
            .fillna("MISSING")
            .astype(str)
        )

        categories = (
            model._Booster
            .pandas_categorical[index]
        )

        dataframe[column] = pd.Categorical(
            dataframe[column],
            categories=categories,
        )

    return dataframe


# ============================================================
# HELPERS
# ============================================================

def get_risk_band(
    probability,
):

    if probability < 0.30:
        return "Low"

    if probability < 0.60:
        return "Medium"

    if probability < 0.80:
        return "High"

    return "Critical"


def get_action(
    probability,
):

    if probability < 0.30:
        return "Approve"

    if probability < 0.60:
        return "Monitor"

    if probability < 0.75:
        return "Manual review"

    return "Escalate for review"


def get_case_type(
    actual,
    probability,
):

    predicted = int(
        probability >= 0.75
    )

    if actual == 1 and predicted == 1:
        return "True Positive"

    if actual == 0 and predicted == 1:
        return "False Positive"

    if actual == 1 and predicted == 0:
        return "False Negative"

    return "True Negative"


def yes_no(
    value,
):

    if pd.isna(value):
        return "Unknown"

    return (
        "Yes"
        if int(value) == 1
        else "No"
    )


def mask_card(
    card_number,
):

    card = str(
        card_number
    )

    if len(card) <= 4:
        return card

    return (
        "•••• •••• •••• "
        + card[-4:]
    )


def safe_number(
    value,
    decimals=2,
):

    if pd.isna(value):
        return "N/A"

    return f"{value:,.{decimals}f}"


def render_dark_kv_table(dataframe):
    rows_html = ""

    for _, row in dataframe.iterrows():
        field = html.escape(str(row["Field"]))
        value = html.escape(str(row["Value"]))

        rows_html += f"""
        <tr>
            <td>{field}</td>
            <td>{value}</td>
        </tr>
        """

    render_html(
        f"""
        <table class="dark-kv-table">
            <thead>
                <tr>
                    <th>Field</th>
                    <th>Value</th>
                </tr>
            </thead>
            <tbody>
                {rows_html}
            </tbody>
        </table>
        """
    )


def render_dark_table(dataframe):
    if dataframe.empty:
        return

    headers_html = "".join(
        f"<th>{html.escape(str(col))}</th>"
        for col in dataframe.columns
    )

    rows_html = ""
    for _, row in dataframe.iterrows():
        rows_html += "<tr>"
        for col in dataframe.columns:
            rows_html += f"<td>{html.escape(str(row[col]))}</td>"
        rows_html += "</tr>"

    render_html(
        f"""
        <div class="table-container">
            <table class="dark-table">
                <thead>
                    <tr>
                        {headers_html}
                    </tr>
                </thead>
                <tbody>
                    {rows_html}
                </tbody>
            </table>
        </div>
        """
    )


def get_transaction(
    transaction_key,
):

    feature_sql = ", ".join(
        f"m.{feature}"
        for feature in FEATURES
    )

    return query_database(
        f"""
        SELECT

            m.transaction_key,

            n.source_transaction_id,
            n.customer_id,
            n.card_number,
            n.device_fingerprint,
            n.ip_address,

            n.transaction_ts,

            {feature_sql},

            m.is_fraud,

            n.prior_customers_on_device,
            n.prior_cards_on_device,
            n.prior_devices_on_card,
            n.prior_merchants_on_device,

            n.new_customer_on_existing_device_flag,
            n.new_card_on_existing_device_flag,
            n.device_shared_by_multiple_customers_flag,
            n.device_shared_by_multiple_cards_flag

        FROM modeling.model_base AS m

        INNER JOIN analytics.network_features AS n
            ON m.transaction_key =
               n.transaction_key

        WHERE m.transaction_key = ?

        LIMIT 1
        """,
        [transaction_key],
    )


def score_transaction(
    transaction,
):

    model_data = prepare_categories(
        transaction[
            FEATURES
        ]
    )

    probability = float(
        model.predict_proba(
            model_data,
            num_iteration=model.best_iteration_,
        )[0, 1]
    )

    contributions = (
        model.booster_
        .predict(
            model_data,
            pred_contrib=True,
            num_iteration=model.best_iteration_,
        )[0]
    )

    shap_values = (
        contributions[:-1]
    )

    rows = []

    for feature, contribution in zip(
        FEATURES,
        shap_values,
    ):

        rows.append(
            {
                "Feature":
                    FRIENDLY_NAMES.get(
                        feature,
                        feature,
                    ),

                "Value":
                    transaction.iloc[
                        0
                    ][
                        feature
                    ],

                "Impact":
                    float(
                        contribution
                    ),

                "Direction":
                    (
                        "Raises risk"
                        if contribution > 0
                        else "Lowers risk"
                    ),
            }
        )

    explanation = pd.DataFrame(
        rows
    )

    explanation[
        "Absolute impact"
    ] = (
        explanation[
            "Impact"
        ].abs()
    )

    explanation = (
        explanation
        .sort_values(
            "Absolute impact",
            ascending=False,
        )
    )

    return (
        probability,
        explanation,
    )


# ============================================================
# SIDEBAR
# ============================================================

render_html(
    """
    <div class="eyebrow">
        Risk Operations
    </div>

    <div style="
        font-size:1.35rem;
        font-weight:600;
        color:#f2eee7;
        margin-bottom:1.4rem;
    ">
        Investigation Queue
    </div>
    """,
    st.sidebar,
)


queue_type = st.sidebar.selectbox(
    "Queue",
    [
        "Highest risk transactions",
        "False positives",
        "False negatives",
        "Random test transactions",
    ],
)


if queue_type == "Highest risk transactions":

    queue = query_database(
        """
        SELECT
            transaction_key,
            fraud_probability,
            is_fraud

        FROM modeling.test_scored

        ORDER BY
            fraud_probability DESC

        LIMIT 100
        """
    )


elif queue_type == "False positives":

    queue = query_database(
        """
        SELECT
            transaction_key,
            fraud_probability,
            is_fraud

        FROM modeling.test_scored

        WHERE
            is_fraud = 0
            AND fraud_probability >= 0.75

        ORDER BY
            fraud_probability DESC

        LIMIT 100
        """
    )


elif queue_type == "False negatives":

    queue = query_database(
        """
        SELECT
            transaction_key,
            fraud_probability,
            is_fraud

        FROM modeling.test_scored

        WHERE
            is_fraud = 1
            AND fraud_probability < 0.75

        ORDER BY
            fraud_probability DESC

        LIMIT 100
        """
    )


else:

    queue = query_database(
        """
        SELECT
            transaction_key,
            fraud_probability,
            is_fraud

        FROM modeling.test_scored

        USING SAMPLE 100 ROWS
        """
    )


transaction_key = st.sidebar.selectbox(
    "Select transaction",
    queue[
        "transaction_key"
    ].tolist(),
)


manual_key = st.sidebar.text_input(
    "Search transaction key",
    placeholder="Enter transaction ID"
)


if manual_key.strip():

    transaction_key = (
        manual_key.strip()
    )


render_html(
    """
    <br>
    <hr>
    """,
    st.sidebar,
)


render_html(
    """
    <div style="
        color:#697585;
        font-size:0.72rem;
        line-height:1.8;
    ">

        DECISION THRESHOLD
        <br>

        <span style="
            color:#c9b37e;
            font-size:0.92rem;
        ">
            75%
        </span>

        <br><br>

        DATA WINDOW
        <br>

        <span style="
            color:#aeb6c2;
        ">
            Sep 30 – Oct 30, 2024
        </span>

        <br><br>

        MODEL
        <br>

        <span style="
            color:#aeb6c2;
        ">
            LightGBM Risk Model
        </span>

    </div>
    """,
    st.sidebar,
)


# ============================================================
# LOAD TRANSACTION
# ============================================================

transaction = get_transaction(
    transaction_key
)


if transaction.empty:

    st.error(
        "Transaction not found."
    )

    st.stop()


probability, explanation = score_transaction(
    transaction
)


row = transaction.iloc[0]


actual_fraud = int(
    row[
        "is_fraud"
    ]
)


risk_score = round(
    probability * 100
)


risk_band = get_risk_band(
    probability
)


action = get_action(
    probability
)


case_type = get_case_type(
    actual_fraud,
    probability,
)


# ============================================================
# HERO
# ============================================================

render_html(
    """
    <div class="eyebrow">
        Payment Risk Intelligence
    </div>

    <div class="hero-title">
        Fraud Investigation
    </div>

    <div class="hero-subtitle">
        Transaction-level risk scoring,
        behavioral intelligence,
        explainable machine learning
        and relationship analysis.
    </div>
    """
)


render_html(
    f"""
    <div class="transaction-strip">

        <span class="gold-text">
            {row['transaction_key']}
        </span>

        <span class="transaction-divider">
            /
        </span>

        <span class="transaction-meta">
            {row['transaction_ts']}
        </span>

        <span class="transaction-status">

            <span class="status-pill status-critical">
                {risk_band} Risk
            </span>

            <span class="status-pill status-neutral">
                {case_type}
            </span>

        </span>

    </div>
    """
)


# ============================================================
# SUMMARY CARDS
# ============================================================

col1, col2, col3, col4 = st.columns(4)


with col1:

    render_html(
        f"""
        <div class="metric-card">

            <div class="metric-label">
                Risk Score
            </div>

            <div class="metric-value critical">

                {risk_score}

                <span style="
                    font-size:1rem;
                    color:#788392;
                ">
                    /100
                </span>

            </div>

            <div class="metric-note">
                Model-derived risk index
            </div>

        </div>
        """
    )


with col2:

    render_html(
        f"""
        <div class="metric-card">

            <div class="metric-label">
                Risk Classification
            </div>

            <div class="metric-value">
                {risk_band}
            </div>

            <div class="metric-note">
                Operational risk band
            </div>

        </div>
        """
    )


with col3:

    render_html(
        f"""
        <div class="metric-card">

            <div class="metric-label">
                Fraud Probability
            </div>

            <div class="metric-value">
                {probability:.2%}
            </div>

            <div class="metric-note">
                LightGBM model probability
            </div>

        </div>
        """
    )


with col4:

    render_html(
        f"""
        <div class="metric-card">

            <div class="metric-label">
                Recommended Action
            </div>

            <div class="metric-value"
                 style="font-size:1.35rem;">
                {action}
            </div>

            <div class="metric-note">
                Project decision policy
            </div>

        </div>
        """
    )


st.write("")


# ============================================================
# TABS
# ============================================================

overview_tab, explanation_tab, behavior_tab, network_tab = st.tabs(
    [
        "Overview",
        "Risk Drivers",
        "Customer Intelligence",
        "Relationship Analysis",
    ]
)


# ============================================================
# OVERVIEW
# ============================================================

with overview_tab:

    render_html(
        """
        <div class="section-title">
            Transaction Profile
        </div>
        """
    )


    d1, d2, d3, d4 = st.columns(4)


    detail_cards = [
        (
            "Amount",
            f"{row['amount']:,.2f} {row['currency']}"
        ),

        (
            "Merchant",
            str(
                row[
                    "merchant"
                ]
            )
        ),

        (
            "Channel",
            str(
                row[
                    "channel"
                ]
            ).upper()
        ),

        (
            "Country",
            str(
                row[
                    "country"
                ]
            )
        ),
    ]


    for column, (
        label,
        value
    ) in zip(
        [
            d1,
            d2,
            d3,
            d4,
        ],
        detail_cards,
    ):

        with column:

            render_html(
                f"""
                <div class="detail-card">

                    <div class="detail-label">
                        {label}
                    </div>

                    <div class="detail-value">
                        {value}
                    </div>

                </div>
                """
            )


    st.write("")


    left, right = st.columns(
        [1.15, 0.85]
    )


    with left:

        render_html(
            """
            <div class="section-title">
                Transaction Details
            </div>
            """
        )


        details = pd.DataFrame(
            {
                "Field": [
                    "Timestamp",
                    "Customer",
                    "Card",
                    "Merchant category",
                    "Merchant type",
                    "Card type",
                    "Device",
                    "Card present",
                    "Distance from home",
                    "Transaction hour",
                ],

                "Value": [
                    row[
                        "transaction_ts"
                    ],

                    row[
                        "customer_id"
                    ],

                    mask_card(
                        row[
                            "card_number"
                        ]
                    ),

                    row[
                        "merchant_category"
                    ],

                    row[
                        "merchant_type"
                    ],

                    row[
                        "card_type"
                    ],

                    row[
                        "device"
                    ],

                    yes_no(
                        row[
                            "card_present"
                        ]
                    ),

                    safe_number(
                        row[
                            "distance_from_home"
                        ]
                    ),

                    row[
                        "transaction_hour"
                    ],
                ],
            }
        )

        # Retained customized kv pair table for this section
        render_dark_kv_table(details)


    with right:

        render_html(
            """
            <div class="section-title">
                Decision Context
            </div>
            """
        )


        ground_truth = (
            "Fraud"
            if actual_fraud == 1
            else "Legitimate"
        )


        render_html(
            f"""
            <div class="insight-box">

                <div style="
                    color:#8d97a5;
                    font-size:0.72rem;
                    letter-spacing:0.08em;
                    text-transform:uppercase;
                    margin-bottom:0.5rem;
                ">
                    Model Decision
                </div>

                <div style="
                    color:#f2eee7;
                    font-size:1.2rem;
                    margin-bottom:1rem;
                ">
                    {case_type}
                </div>

                <div style="
                    color:#8d97a5;
                    font-size:0.72rem;
                    letter-spacing:0.08em;
                    text-transform:uppercase;
                    margin-bottom:0.5rem;
                ">
                    Ground Truth
                </div>

                <div style="
                    color:#dfc98f;
                    font-size:1.2rem;
                    margin-bottom:1rem;
                ">
                    {ground_truth}
                </div>

                <div style="
                    color:#8d97a5;
                    font-size:0.72rem;
                    letter-spacing:0.08em;
                    text-transform:uppercase;
                    margin-bottom:0.5rem;
                ">
                    Decision Threshold
                </div>

                <div style="
                    color:#f2eee7;
                    font-size:1.2rem;
                ">
                    75%
                </div>

            </div>
            """
        )


# ============================================================
# RISK DRIVERS
# ============================================================

with explanation_tab:

    render_html(
        """
        <div class="section-title">
            Model Risk Drivers
        </div>

        <div style="
            color:#8e98a6;
            margin-bottom:1.2rem;
            max-width:850px;
        ">
            SHAP contributions explain which
            transaction characteristics pushed
            this case toward or away from fraud.
        </div>
        """
    )


    top_explanation = (
        explanation
        .head(10)
        .copy()
    )


    plot_data = (
        top_explanation
        .sort_values(
            "Absolute impact",
            ascending=True,
        )
    )


    colors = [
        (
            "#d2b875"
            if value > 0
            else "#64748b"
        )

        for value in plot_data[
            "Impact"
        ]
    ]


    fig = go.Figure()


    fig.add_trace(
        go.Bar(

            x=plot_data[
                "Impact"
            ],

            y=plot_data[
                "Feature"
            ],

            orientation="h",

            marker=dict(
                color=colors,
            ),

            customdata=np.stack(
                (
                    plot_data[
                        "Value"
                    ].astype(str),

                    plot_data[
                        "Direction"
                    ],
                ),
                axis=-1,
            ),

            hovertemplate=(
                "<b>%{y}</b>"
                "<br>Value: %{customdata[0]}"
                "<br>%{customdata[1]}"
                "<br>SHAP contribution: %{x:.3f}"
                "<extra></extra>"
            ),
        )
    )


    fig.add_vline(
        x=0,
        line_width=1,
        line_color=
            "rgba(255,255,255,0.18)",
    )


    fig.update_layout(

        height=520,

        margin=dict(
            l=10,
            r=20,
            t=20,
            b=20,
        ),

        paper_bgcolor=
            "rgba(0,0,0,0)",

        plot_bgcolor=
            "rgba(0,0,0,0)",

        font=dict(
            color="#c9ced7",
        ),

        xaxis=dict(
            title=
                "Contribution to fraud risk",

            showgrid=True,

            gridcolor=
                "rgba(255,255,255,0.05)",

            zeroline=False,
        ),

        yaxis=dict(
            title="",

            tickfont=dict(
                size=12
            ),
        ),

        showlegend=False,
    )


    st.plotly_chart(
        fig,
        use_container_width=True,
    )


    display_explanation = (
        top_explanation[
            [
                "Feature",
                "Value",
                "Direction",
                "Impact",
            ]
        ]
        .copy()
    )


    display_explanation[
        "Impact"
    ] = (
        display_explanation[
            "Impact"
        ].round(
            3
        )
    )


    # Replaced st.dataframe with custom generic table renderer
    render_dark_table(display_explanation)


# ============================================================
# CUSTOMER INTELLIGENCE
# ============================================================

with behavior_tab:

    render_html(
        """
        <div class="section-title">
            Customer Behavioral Intelligence
        </div>
        """
    )


    b1, b2, b3, b4 = st.columns(4)


    behavior_cards = [
        (
            "Prior-hour activity",
            f"{int(row['calculated_txn_count_1h'])} transactions"
        ),

        (
            "Prior-hour spend",
            safe_number(
                row[
                    "calculated_amount_1h"
                ]
            )
        ),

        (
            "Prior 24h activity",
            f"{int(row['calculated_txn_count_24h'])} transactions"
        ),

        (
            "Prior 24h spend",
            safe_number(
                row[
                    "calculated_amount_24h"
                ]
            )
        ),
    ]


    for column, (
        label,
        value
    ) in zip(
        [
            b1,
            b2,
            b3,
            b4,
        ],
        behavior_cards,
    ):

        with column:

            render_html(
                f"""
                <div class="detail-card">

                    <div class="detail-label">
                        {label}
                    </div>

                    <div class="detail-value">
                        {value}
                    </div>

                </div>
                """
            )


    st.write("")


    b5, b6, b7 = st.columns(3)


    history_cards = [
        (
            "Historical average amount",

            safe_number(
                row[
                    "customer_avg_amount_prior"
                ]
            )
        ),

        (
            "Amount vs historical average",

            (
                f"{row['amount_vs_customer_average']:.2f}×"

                if pd.notna(
                    row[
                        "amount_vs_customer_average"
                    ]
                )

                else "N/A"
            )
        ),

        (
            "Minutes since previous transaction",

            (
                f"{row['minutes_since_previous_transaction']:,.1f}"

                if pd.notna(
                    row[
                        "minutes_since_previous_transaction"
                    ]
                )

                else "First transaction"
            )
        ),
    ]


    for column, (
        label,
        value
    ) in zip(
        [
            b5,
            b6,
            b7,
        ],
        history_cards,
    ):

        with column:

            render_html(
                f"""
                <div class="detail-card">

                    <div class="detail-label">
                        {label}
                    </div>

                    <div class="detail-value">
                        {value}
                    </div>

                </div>
                """
            )


    st.write("")


    render_html(
        """
        <div class="section-title">
            Recent Customer Activity
        </div>
        """
    )


    customer_history = query_database(
        """
        SELECT

            transaction_ts
                AS timestamp,

            transaction_key,

            amount,

            currency,

            merchant,

            channel,

            country,

            CASE
                WHEN is_fraud = 1
                THEN 'Fraud'
                ELSE 'Legitimate'
            END AS outcome

        FROM analytics.network_features

        WHERE
            customer_id = ?
            AND transaction_ts <= ?

        ORDER BY
            transaction_ts DESC

        LIMIT 20
        """,
        [
            row[
                "customer_id"
            ],

            row[
                "transaction_ts"
            ],
        ],
    )


    # Replaced st.dataframe with custom generic table renderer
    render_dark_table(customer_history)


# ============================================================
# RELATIONSHIP ANALYSIS
# ============================================================

with network_tab:

    render_html(
        """
        <div class="section-title">
            Relationship Intelligence
        </div>

        <div style="
            color:#8e98a6;
            margin-bottom:1.2rem;
        ">
            Historical card-device-customer
            relationships help investigators
            identify shared infrastructure
            and suspicious account linkage.
        </div>
        """
    )


    n1, n2, n3, n4 = st.columns(4)


    network_cards = [
        (
            "Prior customers on device",

            int(
                row[
                    "prior_customers_on_device"
                ]
            )
        ),

        (
            "Prior cards on device",

            int(
                row[
                    "prior_cards_on_device"
                ]
            )
        ),

        (
            "Prior devices on card",

            int(
                row[
                    "prior_devices_on_card"
                ]
            )
        ),

        (
            "Prior merchants on device",

            int(
                row[
                    "prior_merchants_on_device"
                ]
            )
        ),
    ]


    for column, (
        label,
        value
    ) in zip(
        [
            n1,
            n2,
            n3,
            n4,
        ],
        network_cards,
    ):

        with column:

            render_html(
                f"""
                <div class="detail-card">

                    <div class="detail-label">
                        {label}
                    </div>

                    <div class="detail-value">
                        {value}
                    </div>

                </div>
                """
            )


    st.write("")


    network_flags = pd.DataFrame(
        {
            "Network signal": [
                "New customer on existing device",
                "New card on existing device",
                "Device shared by multiple customers",
                "Device shared by multiple cards",
            ],

            "Status": [
                yes_no(
                    row[
                        "new_customer_on_existing_device_flag"
                    ]
                ),

                yes_no(
                    row[
                        "new_card_on_existing_device_flag"
                    ]
                ),

                yes_no(
                    row[
                        "device_shared_by_multiple_customers_flag"
                    ]
                ),

                yes_no(
                    row[
                        "device_shared_by_multiple_cards_flag"
                    ]
                ),
            ],
        }
    )


    render_html(
        """
        <div class="section-title">
            Network Signals
        </div>
        """
    )


    # Replaced st.dataframe with custom generic table renderer
    render_dark_table(network_flags)


    device_customers = query_database(
        """
        SELECT

            customer_id,

            transactions,

            cards,

            first_seen,

            last_seen,

            fraud_transactions,

            fraud_rate_pct

        FROM analytics.device_customer_edges

        WHERE
            device_fingerprint = ?

        ORDER BY
            fraud_transactions DESC,
            transactions DESC

        LIMIT 50
        """,
        [
            row[
                "device_fingerprint"
            ]
        ],
    )


    device_cards = query_database(
        """
        SELECT

            card_number,

            transactions,

            customers,

            first_seen,

            last_seen,

            fraud_transactions,

            fraud_rate_pct

        FROM analytics.device_card_edges

        WHERE
            device_fingerprint = ?

        ORDER BY
            fraud_transactions DESC,
            transactions DESC

        LIMIT 50
        """,
        [
            row[
                "device_fingerprint"
            ]
        ],
    )


    if not device_cards.empty:

        device_cards[
            "card_number"
        ] = (
            device_cards[
                "card_number"
            ]
            .apply(
                mask_card
            )
        )


    customer_tab, card_tab = st.tabs(
        [
            "Linked customers",
            "Linked cards",
        ]
    )


    with customer_tab:
        
        # Replaced st.dataframe with custom generic table renderer
        render_dark_table(device_customers)


    with card_tab:

        # Replaced st.dataframe with custom generic table renderer
        render_dark_table(device_cards)


# ============================================================
# FOOTER
# ============================================================

render_html(
    """
    <div class="footer-note">

        Portfolio demonstration built on synthetic payment
        transaction data. Risk classifications and recommended
        actions are analytical project rules and do not
        represent real banking policy.

    </div>
    """
)