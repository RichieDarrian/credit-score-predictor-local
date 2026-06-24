import streamlit as st
import joblib
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(
    page_title="Credit Score Predictor",
    layout="wide",
    initial_sidebar_state="collapsed",
)


@st.cache_resource
def load_model():
    return joblib.load("artifacts/random_forest.pkl")

model = load_model()

LABEL_MAP = {0: "Poor", 1: "Standard", 2: "Good"}

PAYMENT_BEHAVIOUR_LABELS = {
    "High_spent_Small_value_payments":  "High Spending - Small Transactions",
    "Low_spent_Large_value_payments":   "Low Spending - Large Transactions",
    "Low_spent_Medium_value_payments":  "Low Spending - Medium Transactions",
    "Low_spent_Small_value_payments":   "Low Spending - Small Transactions",
    "High_spent_Medium_value_payments": "High Spending - Medium Transactions",
    "High_spent_Large_value_payments":  "High Spending - Large Transactions",
}

PAYMENT_BEHAVIOUR_REVERSE = {v: k for k, v in PAYMENT_BEHAVIOUR_LABELS.items()}

LOAN_COLS = [
    "Auto Loan",
    "Credit-Builder Loan",
    "Debt Consolidation Loan",
    "Home Equity Loan",
    "Mortgage Loan",
    "Payday Loan",
    "Personal Loan",
    "Student Loan",
]


def build_input_df(inputs: dict) -> pd.DataFrame:
    loan_flags = {col: (col in inputs["loan_types"]) for col in LOAN_COLS}
    row = {
        "Credit_Utilization_Ratio": inputs["credit_utilization_ratio"],
        "Credit_History_Age": inputs["credit_history_age"],
        "Annual_Income": inputs["annual_income"],
        "Monthly_Inhand_Salary": inputs["monthly_inhand_salary"],
        "Num_Bank_Accounts": inputs["num_bank_accounts"],
        "Num_Credit_Card": inputs["num_credit_card"],
        "Interest_Rate": inputs["interest_rate"],
        "Num_of_Loan": inputs["num_of_loan"],
        "Delay_from_due_date": inputs["delay_from_due_date"],
        "Num_of_Delayed_Payment": inputs["num_of_delayed_payment"],
        "Changed_Credit_Limit": inputs["changed_credit_limit"],
        "Num_Credit_Inquiries": inputs["num_credit_inquiries"],
        "Outstanding_Debt": inputs["outstanding_debt"],
        "Total_EMI_per_month": inputs["total_emi_per_month"],
        "Amount_invested_monthly": inputs["amount_invested_monthly"],
        "Monthly_Balance": inputs["monthly_balance"],
        "Credit_Mix": inputs["credit_mix"],
        "Payment_of_Min_Amount": inputs["payment_of_min_amount"],
        "Payment_Behaviour": inputs["payment_behaviour"],
        **loan_flags,
    }
    return pd.DataFrame([row])


def make_probability_chart(probs: list) -> go.Figure:
    labels = ["Poor", "Standard", "Good"]
    colors = ["#e05a6a", "#f0a500", "#3ecf8e"]

    fig = go.Figure(go.Bar(
        x=[p * 100 for p in probs],
        y=labels,
        orientation="h",
        marker=dict(color=colors, opacity=0.85),
        text=[f"{p*100:.1f}%" for p in probs],
        textposition="outside",
        textfont=dict(size=12),
        hoverinfo="none",
    ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        margin=dict(l=0, r=60, t=8, b=8),
        height=180,
        xaxis=dict(range=[0, 110], ticksuffix="%", gridcolor="#e0e0e0", zeroline=False),
        yaxis=dict(gridcolor="rgba(0,0,0,0)"),
        bargap=0.4,
    )
    return fig


st.title("Credit Score Predictor")

with st.form("prediction_form"):

    st.subheader("Income & Debt")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        annual_income = st.number_input("Annual Income ($)", min_value=0.0, max_value=25000000.0, value=50000.0, step=1000.0)
    with c2:
        monthly_inhand_salary = st.number_input("Monthly Inhand Salary ($)", min_value=0.0, max_value=16000.0, value=4000.0, step=100.0)
    with c3:
        outstanding_debt = st.number_input("Outstanding Debt ($)", min_value=0.0, max_value=5000.0, value=500.0, step=50.0)
    with c4:
        changed_credit_limit = st.number_input("Changed Credit Limit (%)", min_value=-10.0, max_value=40.0, value=0.0, step=0.5)

    c5, c6, c7, c8 = st.columns(4)
    with c5:
        total_emi_per_month = st.number_input("Total EMI per Month ($)", min_value=0.0, max_value=85000.0, value=50.0, step=10.0)
    with c6:
        amount_invested_monthly = st.number_input("Amount Invested Monthly ($)", min_value=0.0, max_value=2000.0, value=100.0, step=10.0)
    with c7:
        monthly_balance = st.number_input("Monthly Balance ($)", min_value=0.0, max_value=1600.0, value=300.0, step=50.0)
    with c8:
        interest_rate = st.number_input("Interest Rate (%)", min_value=0, max_value=6000, value=15, step=1)

    st.subheader("Account Activity")
    c9, c10, c11, c12 = st.columns(4)
    with c9:
        num_bank_accounts = st.number_input("Number of Bank Accounts", min_value=0, max_value=1800, value=3, step=1)
    with c10:
        num_credit_card = st.number_input("Number of Credit Cards", min_value=0, max_value=1500, value=2, step=1)
    with c11:
        num_of_loan = st.number_input("Number of Loans", min_value=0, max_value=1500, value=2, step=1)
    with c12:
        num_credit_inquiries = st.number_input("Number of Credit Inquiries", min_value=0, max_value=2600, value=3, step=1)

    c13, c14, c15, c16 = st.columns(4)
    with c13:
        num_of_delayed_payment = st.number_input("Number of Delayed Payments", min_value=0, max_value=4400, value=5, step=1)
    with c14:
        delay_from_due_date = st.number_input("Avg Delay from Due Date (days)", min_value=-10, max_value=70, value=10, step=1)
    with c15:
        credit_history_age = st.number_input("Credit History Age (months)", min_value=0, max_value=410, value=180, step=1)
    with c16:
        credit_utilization_ratio = st.number_input("Credit Utilization Ratio (%)", min_value=0.0, max_value=50.0, value=30.0, step=0.5)

    st.subheader("Credit Profile")
    c17, c18, c19 = st.columns(3)
    with c17:
        credit_mix = st.selectbox("Credit Mix", options=["Good", "Standard", "Bad", "Unknown"])
    with c18:
        payment_of_min_amount = st.selectbox("Payment of Min Amount", options=["Yes", "No", "Unknown"])
    with c19:
        payment_behaviour_label = st.selectbox(
            "Payment Behaviour",
            options=list(PAYMENT_BEHAVIOUR_LABELS.values()),
            help="How you typically spend and what transaction sizes you make most often.",
        )
        payment_behaviour = PAYMENT_BEHAVIOUR_REVERSE[payment_behaviour_label]

    st.subheader("Active Loan Types")
    loan_types = st.multiselect(
        "Pilih semua jenis pinjaman yang aktif (boleh kosong)",
        options=LOAN_COLS,
    )

    submitted = st.form_submit_button("Jalankan Prediksi", use_container_width=True, type="primary")

if submitted:
    inputs = {
        "annual_income": annual_income,
        "monthly_inhand_salary": monthly_inhand_salary,
        "outstanding_debt": outstanding_debt,
        "total_emi_per_month": total_emi_per_month,
        "amount_invested_monthly": amount_invested_monthly,
        "monthly_balance": monthly_balance,
        "changed_credit_limit": changed_credit_limit,
        "num_bank_accounts": num_bank_accounts,
        "num_credit_card": num_credit_card,
        "num_of_loan": num_of_loan,
        "num_of_delayed_payment": num_of_delayed_payment,
        "num_credit_inquiries": num_credit_inquiries,
        "interest_rate": interest_rate,
        "delay_from_due_date": delay_from_due_date,
        "credit_history_age": credit_history_age,
        "credit_utilization_ratio": credit_utilization_ratio,
        "credit_mix": credit_mix,
        "payment_of_min_amount": payment_of_min_amount,
        "payment_behaviour": payment_behaviour,
        "loan_types": loan_types,
    }

    df_input = build_input_df(inputs)
    prediction_code = int(model.predict(df_input)[0])
    prediction_label = LABEL_MAP[prediction_code]
    probs = model.predict_proba(df_input)[0].tolist()

    st.divider()

    if prediction_label == "Good":
        st.success(f"Credit Score: **{prediction_label}**")
    elif prediction_label == "Standard":
        st.warning(f"Credit Score: **{prediction_label}**")
    else:
        st.error(f"Credit Score: **{prediction_label}**")

    st.subheader("Probability Distribution")
    st.plotly_chart(make_probability_chart(probs), use_container_width=True, config={"displayModeBar": False})