import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ---------- Helper: Predict Category ----------
def predict_category(description: str) -> str:
    """Predict category using OpenAI, fallback to simple rules if API fails."""
    prompt = f"""
    You are a financial assistant. Categorize this expense into one of:
    ['Food', 'Transportation', 'Entertainment', 'Utilities', 'Shopping', 'Others'].

    Expense: "{description}"
    Just return the category name.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        return response.choices[0].message.content.strip()
    except Exception:
        # --- Fallback: simple keyword-based classification ---
        desc = description.lower()
        if any(word in desc for word in ["food", "restaurant", "shawarma", "pizza"]):
            return "Food"
        elif any(word in desc for word in ["uber", "bus", "taxi", "transport"]):
            return "Transportation"
        elif any(word in desc for word in ["movie", "game", "netflix", "music"]):
            return "Entertainment"
        elif any(word in desc for word in ["electricity", "water", "internet", "bill"]):
            return "Utilities"
        elif any(word in desc for word in ["mall", "clothes", "shop", "buy"]):
            return "Shopping"
        return "Others"

# ---------- Data Handling ----------
csv_file = "expense_data_1.csv"
if os.path.exists(csv_file):
    data = pd.read_csv(csv_file)
else:
    data = pd.DataFrame(columns=["Date", "Description", "Amount", "Category"])

# ---------- Streamlit UI ----------
st.title(" Smart Expense Tracker")

with st.form("expense_form"):
    date = st.date_input("Date")
    description = st.text_input("Description")
    amount = st.number_input("Amount", min_value=0.0, format="%.2f")

    predicted_category = ""
    if description:
        predicted_category = predict_category(description)

    category = st.text_input(
        "Category (auto-predicted, but you can edit)", 
        value=predicted_category
    )

    submitted = st.form_submit_button("Add Expense")

    if submitted:
        new_expense = {
            "Date": str(date),
            "Description": description,
            "Amount": amount,
            "Category": category
        }
        data = pd.concat([data, pd.DataFrame([new_expense])], ignore_index=True)
        data.to_csv(csv_file, index=False)
        st.success(f" Added: {description} - {amount} ({category})")

# ---------- Display All Expenses ----------
st.subheader("All Expenses")
st.dataframe(data)

# ---------- Charts ----------
if not data.empty:
    st.subheader(" Expense Breakdown by Category")

    category_totals = data.groupby("Category")["Amount"].sum()

    # Bar Chart
    fig, ax = plt.subplots()
    category_totals.plot(kind="bar", ax=ax, color="skyblue")
    ax.set_ylabel("Amount")
    st.pyplot(fig)

    # Pie Chart
    st.subheader(" Category Distribution")
    fig2, ax2 = plt.subplots()
    category_totals.plot(kind="pie", autopct="%1.1f%%", ax=ax2)
    ax2.set_ylabel("")  # remove y-axis label for pie
    st.pyplot(fig2)
