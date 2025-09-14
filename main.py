import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
import os
import matplotlib.pyplot as plt

# Load environment variables
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
print(f"Client Loaded: {client is not None}")

# Load expense dataset
df = pd.read_csv("./datasets/expense_data_1.csv")

# Ensure required columns exist
required_cols = ["Date", "Category", "Note", "Amount", "Income/Expense"]
for col in required_cols:
    if col not in df.columns:
        df[col] = None  # add missing columns if not in CSV

data = df[required_cols]
print("Initial Data Preview:")
print(data.head())

# Add new expense
def add_expense(date, category, note, amount, exp_type="Expense"):
    global data
    new_entry = {
        "Date": date,
        "Category": category,
        "Note": note,
        "Amount": amount,
        "Income/Expense": exp_type
    }
    # ✅ Use pd.concat instead of append
    data = pd.concat([data, pd.DataFrame([new_entry])], ignore_index=True)
    print(f"Added: {note} - {amount} ({category}) on {date} as {exp_type}")

# Example expenses
add_expense("2025-08-22 19:30", "Food", "Shawarma", 2500, "Expense")
add_expense("2025-08-23 08:00", "Subscriptions", "Netflix Monthly Plan", 4500, "Expense")
add_expense("2025-08-24 14:00", "Entertainment", "Outdoor Games with friends", 7000, "Expense")

# View latest expenses
def view_expenses(n=10):
    return data.tail(n)

print("\nRecent Expenses:")
print(view_expenses(10))

# Summarize expenses
def summarize_expenses(by="Category"):
    # Filter only expenses
    summary = data[data["Income/Expense"] == "Expense"].groupby(by)["Amount"].sum()
    return summary.sort_values(ascending=False)

print("\nExpense Summary by Category:")
print(summarize_expenses())

# Auto Categorization with OpenAI
def auto_categorize(note):
    prompt = f"""
    Categorize the following expense note into one of these categories: 
    Food, Transport, Entertainment, Utilities, Subscriptions, Health, Education, Shopping, Others.
    Note: {note}
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return "Others"

# Apply categorization only if Category is missing/empty
data["Category"] = data.apply(
    lambda row: auto_categorize(row["Note"]) if pd.isna(row["Category"]) or row["Category"] == "" else row["Category"],
    axis=1
)

print("\nAuto-categorization complete.")
print(data[["Note", "Category"]].head(10))

# Visualize expenses
expense_summary = data[data['Category'] != "Income"].groupby('Category')['Amount'].sum()

# Plotting pie chart
plt.figure(figsize=(10, 6))
expense_summary.plot.pie(autopct='%1.1f%%', startangle=140)
plt.title('Expense Distribution by Category')
plt.ylabel('')
plt.savefig('expense_distribution.png')
plt.close()
print("\nExpense distribution chart saved as 'expense_distribution.png'.")

# plotting bie chart
plt.figure(figsize=(10, 6))
expense_summary.plot.pie(autopct='%1.1f%%', startangle=140)
plt.title('Expense Distribution by Category')
plt.ylabel('')
plt.savefig('expense_distribution.png')
plt.close()
print("\nExpense distribution chart saved as 'expense_distribution.png'.")