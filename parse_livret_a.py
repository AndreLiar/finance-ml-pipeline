"""
Livret A Statement Parser
=========================
Parses Livret A (savings account 24971411768) annual statements.

Format: plain text with 3 date columns + LIBELLE + MONTANT
  - Positive amounts = deposits INTO Livret A (salary/savings)
  - Negative amounts = transfers OUT to Compte Cheques (withdrawals)

Output:
  data/livret_a_transactions.xlsx  — Transactions + Monthly Summary
  data/merged_transactions.xlsx    — Combined Compte Cheques + Livret A (income-aware)
"""

import re
import pandas as pd
import numpy as np
import pdfplumber
from pathlib import Path

from config import (
    LIVRET_A_DIR,
    TRANSACTIONS_XLSX  as CHEQUES_DATA,
    LIVRET_A_XLSX      as OUTPUT_LIVRET,
    MERGED_XLSX        as OUTPUT_MERGED,
)
from db import read_table, write_table, table_exists

print("=" * 60)
print("LIVRET A STATEMENT PARSER")
print("=" * 60)

# ── 1. PARSE LIVRET A PDFs ─────────────────────────────────────────────────────

# Regex: DD/MM  LIBELLE  ref  DD/MM  DD/MM  [- ]amount  (euro sign stripped)
TX_RE = re.compile(
    r'^(\d{2}/\d{2})\s+(.+?)\s+(\d{2}/\d{2})\s+(\d{2}/\d{2})\s+([-\d\s,]+)\s*[€\xe2]',
    re.UNICODE
)
# Simpler pattern for single-line transaction: date  label ref  date  date  amount
# The actual format is: DD/MM  LIBELLE REF  DD/MM  DD/MM  [- ]N NNN,NN €
TX_SIMPLE = re.compile(
    r'^(\d{2}/\d{2})\s+(.+?)\s+(\d{2}/\d{2})\s+(\d{2}/\d{2})\s+([-\d\s,]+)$'
)

SKIP_KEYWORDS = [
    'DATE', 'LIBELLE', 'MONTANT', 'COMPTA', 'OPERATION', 'VALEUR',
    'SOLDE CREDITEUR', 'TOTAL DES', 'PAGE', 'Page', 'JE CONSERVE',
    'CE DOCUMENT', 'Banque Populaire', 'sous le N', 'Votre',
    'VOTRE LIVRET', 'DETAIL DES', 'M KANMEGNE', 'BIC', 'IBAN',
    'Garantie', 'Information', 'Simple', 'Application', 'son expertise',
    'utiles', 'mediateur', 'formulaire', 'sous reserve', 'Ce document',
]

def parse_amount(s: str) -> float:
    """Convert '- 1 610,00' or '4 724,17' to float."""
    s = s.strip()
    negative = s.startswith('-')
    s = s.replace('-', '').replace(' ', '').replace(',', '.').strip()
    try:
        val = float(s)
        return -val if negative else val
    except ValueError:
        return None

def year_from_filename(fname: str) -> int:
    """Extract year from filename like '20250107-...'"""
    # The date prefix is 8 digits starting with 20, e.g. 20250107
    m = re.search(r'(?<!\d)(20\d{2})(\d{4})(?=-)', fname)
    return int(m.group(1)) if m else None

def parse_livret_pdf(pdf_path: Path) -> list[dict]:
    """Parse one Livret A annual statement PDF."""
    year = year_from_filename(pdf_path.name)
    print(f"\n  Parsing: {pdf_path.name[:60]}")
    print(f"  Year: {year}")

    transactions = []
    prev_year = year - 1  # Jan statement covers Sep-Dec of previous year

    with pdfplumber.open(pdf_path) as pdf:
        lines = []
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                lines.extend(text.split('\n'))

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Skip non-transaction lines
        skip = False
        for kw in SKIP_KEYWORDS:
            if kw.lower() in line.lower():
                skip = True
                break
        if skip or not line:
            i += 1
            continue

        # Try to match transaction line: starts with DD/MM
        m = re.match(r'^(\d{2}/\d{2})\s+(.*)', line)
        if m:
            date_str = m.group(1)
            rest = m.group(2).strip()

            # The rest may end with the amount, or amount is on same line
            # Pattern: LIBELLE REF  DD/MM  DD/MM  AMOUNT
            # Try full pattern first
            full_m = re.match(r'^(.+?)\s+\d{7}\s+(\d{2}/\d{2})\s+(\d{2}/\d{2})\s+([-\d\s,]+)', rest)
            if not full_m:
                # Try without ref number
                full_m = re.match(r'^(.+?)\s+(\d{2}/\d{2})\s+(\d{2}/\d{2})\s+([-\d\s,]+)', rest)

            if full_m:
                # Get label and amount
                label = full_m.group(1).strip()
                amount_str = full_m.group(full_m.lastindex).strip()
                amount = parse_amount(amount_str)

                # Check if next line is a continuation (e.g. "Virement vers Livret A Particuli")
                description = label
                if i + 1 < len(lines):
                    next_line = lines[i + 1].strip()
                    if next_line and not re.match(r'^\d{2}/\d{2}', next_line):
                        skip_next = any(kw.lower() in next_line.lower() for kw in SKIP_KEYWORDS)
                        if not skip_next and len(next_line) < 60:
                            description = label + ' | ' + next_line
                            i += 1

                # Determine year: Jan statements cover Sep prev_year to Jan this_year
                day, month = int(date_str[:2]), int(date_str[3:5])
                if month >= 9:
                    tx_year = prev_year
                else:
                    tx_year = year

                # Handle INTERETS (interest) row
                if 'INTERETS' in label.upper():
                    description = 'INTERETS LIVRET A'

                if amount is not None:
                    transactions.append({
                        'date': f"{tx_year}-{month:02d}-{day:02d}",
                        'description': description,
                        'label': label,
                        'amount': amount,
                        'type': 'CREDIT' if amount > 0 else 'DEBIT',
                        'account': 'Livret A 24971411768',
                    })

        i += 1

    print(f"  Found {len(transactions)} transactions")
    return transactions

# Parse both PDFs
pdf_files = sorted(LIVRET_A_DIR.glob("Extrait De Compte*.pdf"))
all_transactions = []
for pdf_path in pdf_files:
    txs = parse_livret_pdf(pdf_path)
    all_transactions.extend(txs)

livret_df = pd.DataFrame(all_transactions)
livret_df['date'] = pd.to_datetime(livret_df['date'])
livret_df = livret_df.sort_values('date').reset_index(drop=True)
livret_df['year_month'] = livret_df['date'].dt.to_period('M').astype(str)

print(f"\nTotal Livret A transactions: {len(livret_df)}")
print(f"Date range: {livret_df['date'].min().date()} to {livret_df['date'].max().date()}")
print(f"\nCredits (deposits into Livret A)  : {(livret_df['amount'] > 0).sum()} txs | Total: EUR{livret_df[livret_df['amount'] > 0]['amount'].sum():,.2f}")
print(f"Debits  (transfers to Compte Chq) : {(livret_df['amount'] < 0).sum()} txs | Total: EUR{livret_df[livret_df['amount'] < 0]['amount'].sum():,.2f}")

# ── 2. MONTHLY LIVRET A SUMMARY ────────────────────────────────────────────────

monthly_livret = livret_df.groupby('year_month').agg(
    credits=('amount', lambda x: x[x > 0].sum()),
    debits=('amount', lambda x: x[x < 0].sum()),
    net=('amount', 'sum'),
    tx_count=('amount', 'count'),
).reset_index()

# The "income" from Livret A perspective:
# - Credits into Livret A = salary saved (real income received)
# - Debits from Livret A = money sent to Compte Cheques (funding daily expenses)
monthly_livret['transfers_to_checking'] = monthly_livret['debits'].abs()  # cash flowing INTO Compte Cheques
monthly_livret['savings_deposits'] = monthly_livret['credits']             # salary deposited into Livret A

print(f"\nMonthly Livret A summary:")
print(monthly_livret[['year_month','savings_deposits','transfers_to_checking','net']].to_string(index=False))

# ── 3. LOAD COMPTE CHEQUES DATA & MERGE ───────────────────────────────────────

print("\n" + "-" * 60)
print("MERGING WITH COMPTE CHEQUES")
print("-" * 60)

if table_exists("transactions"):
    cheques_df = read_table("transactions", parse_dates=["date_operation"])
else:
    cheques_df = pd.read_excel(CHEQUES_DATA, sheet_name='Transactions')
    cheques_df['date_operation'] = pd.to_datetime(cheques_df['date_operation'])
cheques_df['year_month'] = cheques_df['date_operation'].dt.to_period('M').astype(str)

# Monthly Compte Cheques summary
monthly_cheques = cheques_df.groupby('year_month').agg(
    spend=('debit', 'sum'),
    income_checking=('credit', 'sum'),
    tx_count=('debit', 'count'),
).reset_index()

# ── 4. BUILD UNIFIED MONTHLY VIEW ─────────────────────────────────────────────

all_months = sorted(set(monthly_cheques['year_month'].tolist() + monthly_livret['year_month'].tolist()))
unified = pd.DataFrame({'year_month': all_months})

unified = unified.merge(monthly_cheques[['year_month', 'spend', 'income_checking', 'tx_count']],
                        on='year_month', how='left').fillna(0)
unified = unified.merge(monthly_livret[['year_month', 'savings_deposits', 'transfers_to_checking', 'net']],
                        on='year_month', how='left').fillna(0)

# True income = salary deposits into Livret A + any direct credits to Compte Cheques
# (excluding Livret A transfers which are internal)
unified['total_income'] = unified['savings_deposits'] + unified['income_checking']
unified['total_spend']  = unified['spend']

# True savings = deposits into Livret A minus withdrawals from Livret A
unified['net_savings']   = unified['net']    # Livret A net flow
unified['total_net']     = unified['total_income'] - unified['total_spend']
unified['savings_rate']  = np.where(
    unified['total_income'] > 0,
    (unified['savings_deposits'] / unified['total_income']) * 100,
    0
)

# Filter to months with activity
unified = unified[unified['year_month'] >= '2024-09']  # Livret A opened Sep 2024

print(f"\nUnified monthly view ({len(unified)} months with Livret A data):")
print(unified[['year_month','total_income','total_spend','net_savings','savings_rate','transfers_to_checking']].to_string(index=False))

# ── 5. ENRICHED TRANSACTIONS — TAG LIVRET A TRANSFERS IN COMPTE CHEQUES ────────

# In the Compte Cheques, Livret A deposits appear as CREDIT rows
# Let's identify them and tag correctly
# The transfers FROM Livret A to Compte Cheques show as credits in Compte Cheques
# labeled "VIR M ANDRE KANMEGNE" (same self-transfer pattern)

# Add a source_account column to cheques transactions
cheques_df['source_account'] = 'Compte Cheques 23192700536'
cheques_df['real_income'] = 0.0  # will be enriched

# Mark months where Livret A funded the checking account
livret_transfers = livret_df[livret_df['amount'] < 0].copy()
livret_transfers['transfer_date'] = livret_transfers['date']
livret_transfers['transfer_amount'] = livret_transfers['amount'].abs()

# ── 6. SAVE ──────────────────────────────────────────────────────────────────

with pd.ExcelWriter(OUTPUT_LIVRET, engine='openpyxl') as writer:
    livret_df.to_excel(writer, sheet_name='Livret A Transactions', index=False)
    monthly_livret.to_excel(writer, sheet_name='Monthly Livret A', index=False)

print(f"\nLivret A data saved to: {OUTPUT_LIVRET}")

with pd.ExcelWriter(OUTPUT_MERGED, engine='openpyxl') as writer:
    unified.to_excel(writer, sheet_name='Unified Monthly', index=False)
    monthly_cheques.to_excel(writer, sheet_name='Compte Cheques Monthly', index=False)
    monthly_livret.to_excel(writer, sheet_name='Livret A Monthly', index=False)
    livret_df.to_excel(writer, sheet_name='Livret A Transactions', index=False)

# ── Write to SQLite ────────────────────────────────────────────────────────────
write_table(livret_df,       "livret_a")
write_table(monthly_livret,  "livret_a_monthly")
write_table(unified,         "unified_monthly")

print(f"Merged data saved to: {OUTPUT_MERGED}")

# ── 7. SUMMARY ────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("FINANCIAL PICTURE — COMPTE CHEQUES ONLY vs COMBINED")
print("=" * 60)

overlap = unified[unified['year_month'] >= '2024-09']
print(f"\nPeriod: Sep 2024 – present ({len(overlap)} months with Livret A)")
print(f"\nCompte Cheques only view:")
print(f"  Avg monthly income : EUR{overlap['income_checking'].mean():,.0f}")
print(f"  Avg monthly spend  : EUR{overlap['total_spend'].mean():,.0f}")
print(f"  Avg savings rate   : {(overlap['income_checking'] / overlap['total_spend'].replace(0,1)).mean()*100:.0f}% (misleading)")

print(f"\nCombined (true) view:")
print(f"  Avg salary to Livret A : EUR{overlap['savings_deposits'].mean():,.0f}/month")
print(f"  Avg transfers to chq   : EUR{overlap['transfers_to_checking'].mean():,.0f}/month")
print(f"  Avg total spend        : EUR{overlap['total_spend'].mean():,.0f}/month")
print(f"  Avg true savings rate  : {overlap['savings_rate'].mean():.1f}%")
print(f"  Avg monthly net        : EUR{overlap['total_net'].mean():,.0f}")

print(f"\nLivret A balance at end of 2025: EUR9,268.57")
print(f"This represents real accumulated savings!")
