"""
PDF Bank Statement Parser
Banque Populaire Rives de Paris - Compte Cheques 23192700536
Extracts transactions into a structured Excel file for ML pipeline.
"""

import pdfplumber
import pandas as pd
import re
import os
from pathlib import Path

from src.config import STATEMENTS_DIR, TRANSACTIONS_XLSX as OUTPUT_EXCEL
from src.db import write_table
from src.logger import get_logger

log = get_logger(__name__)

# ── Helpers ───────────────────────────────────────────────────────────────────

# Matches lines like: "18/10  COTIS CRISTAL CONFORT  0009328  15/10  15/10  4,95"
# or credit:          "04/11  VIREMENT LC10003  04/11  04/11  1 600,00"
TRANSACTION_RE = re.compile(
    r'^(\d{2}/\d{2})\s+'        # DATE COMPTA  (e.g. 18/10)
    r'(.+?)\s+'                  # LIBELLE (description)
    r'(\d{2}/\d{2})\s+'         # DATE OPERATION
    r'(\d{2}/\d{2})\s+'         # DATE VALEUR
    r'([\d\s]+,\d{2})?'         # DEBIT (optional)
    r'\s*([\d\s]+,\d{2})?$'     # CREDIT (optional)
)

BALANCE_RE = re.compile(
    r'SOLDE (CREDITEUR|DEBITEUR) AU (\d{2}/\d{2}/\d{4})\*?\s+([\d\s]+,\d{2})',
    re.IGNORECASE
)

def extract_year_from_filename(filename: str) -> str:
    """Pull year from filename like '...20221108...' -> '2022'"""
    match = re.search(r'(\d{4})\d{4}-', filename)
    return match.group(1) if match else "2022"

def split_cell(cell):
    """Split a newline-packed cell into a list of values."""
    if not cell:
        return []
    return [v.strip() for v in cell.split('\n') if v.strip()]

def parse_amount(text):
    """Convert French number string '1 600,00' or '-  9,90' -> float."""
    if not text:
        return None
    cleaned = text.replace('\u20ac', '').replace(' ', '').replace(',', '.').strip()
    try:
        return float(cleaned)
    except ValueError:
        return None

# ── Format 1: Old table-based PDFs (2022-2023) ────────────────────────────────

def parse_table_format(pdf_path: Path, year: str) -> list[dict]:
    """Parse old-style PDFs where the table has merged cells with \\n separators."""
    transactions = []

    def full_date(d):
        return f"{d}/{year}" if re.match(r'^\d{2}/\d{2}$', d) else d

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            for table in page.extract_tables():
                for row in table:
                    if not row or len(row) < 4:
                        continue

                    col0 = split_cell(row[0])
                    col1 = split_cell(row[1])
                    col2 = split_cell(row[2])
                    col3 = split_cell(row[3])
                    col4 = split_cell(row[4] if len(row) > 4 else '')
                    col5 = split_cell(row[5] if len(row) > 5 else '')

                    date_entries = [d for d in col0 if re.match(r'^\d{2}/\d{2}$', d)]
                    if not date_entries:
                        continue

                    desc_lines = [l for l in col1
                                  if not re.match(r'^SOLDE\s', l, re.I)
                                  and not re.match(r'^TOTAL\s', l, re.I)
                                  and not re.match(r'^\d{2}/\d{2}/\d{4}', l)]

                    op_dates  = [d for d in col2 if re.match(r'^\d{2}/\d{2}$', d)]
                    val_dates = [d for d in col3 if re.match(r'^\d{2}/\d{2}$', d)]

                    amount_re = re.compile(r'^[\d\s]+,\d{2}$')
                    debits  = [v for v in col4 if amount_re.match(v)]
                    credits = [v for v in col5 if amount_re.match(v)]

                    n = len(date_entries)
                    for i in range(n):
                        date_c = date_entries[i]
                        date_o = op_dates[i]  if i < len(op_dates)  else date_c
                        date_v = val_dates[i] if i < len(val_dates) else date_c

                        lines_per_tx = max(1, len(desc_lines) // n)
                        start = i * lines_per_tx
                        end   = start + lines_per_tx if i < n - 1 else len(desc_lines)
                        desc  = ' | '.join(desc_lines[start:end]) if desc_lines else ''

                        debit_raw  = debits[i]  if i < len(debits)  else None
                        credit_raw = credits[i] if i < len(credits) else None

                        debit  = parse_amount(debit_raw)
                        credit = parse_amount(credit_raw)

                        if credit and not debit:
                            tx_type, amount = 'CREDIT', credit
                        elif debit and not credit:
                            tx_type, amount = 'DEBIT', -debit
                        elif debit and credit:
                            tx_type, amount = 'BOTH', credit - debit
                        else:
                            tx_type, amount = 'UNKNOWN', 0.0

                        transactions.append({
                            'date_compta':    full_date(date_c),
                            'date_operation': full_date(date_o),
                            'date_valeur':    full_date(date_v),
                            'description':    desc,
                            'debit':          debit,
                            'credit':         credit,
                            'amount':         amount,
                            'type':           tx_type,
                            'source_file':    pdf_path.name,
                        })
    return transactions

# ── Format 2: New text-based PDFs (2023+) ─────────────────────────────────────
# Lines look like:
#   08/11 071123 CB****2015 ECNJY6I 08/11 08/11 - 9,90 €
#   followed by description lines like:  TSB NANTERRE 92NANTERRE

TX_LINE_RE = re.compile(
    r'^(\d{2}/\d{2})\s+'           # DATE COMPTA
    r'(.+?)\s+'                     # LIBELLE start
    r'(\d{2}/\d{2})\s+'            # DATE OPERATION
    r'(\d{2}/\d{2})\s+'            # DATE VALEUR
    r'([+-])?\s*([\d\s]+,\d{2})'   # optional sign + MONTANT
)

def parse_text_format(pdf_path: Path, year: str) -> list[dict]:
    """Parse new-style PDFs where each transaction is a plain text line."""
    transactions = []

    def full_date(d):
        return f"{d}/{year}" if re.match(r'^\d{2}/\d{2}$', d) else d

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ''
            lines = text.splitlines()
            i = 0
            while i < len(lines):
                line = lines[i].strip()
                m = TX_LINE_RE.match(line)
                if m:
                    date_c  = m.group(1)
                    desc    = m.group(2).strip()
                    date_o  = m.group(3)
                    date_v  = m.group(4)
                    sign    = m.group(5)
                    montant = m.group(6)

                    # Collect continuation description lines
                    j = i + 1
                    while j < len(lines):
                        next_line = lines[j].strip()
                        # Stop if next line looks like a new transaction or balance
                        if (TX_LINE_RE.match(next_line)
                                or re.match(r'^SOLDE\s', next_line, re.I)
                                or re.match(r'^TOTAL\s', next_line, re.I)
                                or re.match(r'^RECAPITULATIF', next_line, re.I)
                                or re.match(r'^\d{2}/\d{2}\s', next_line)):
                            break
                        if next_line:
                            desc += ' ' + next_line
                        j += 1

                    amount_val = parse_amount(montant)
                    if amount_val is None:
                        i += 1
                        continue

                    if sign == '-':
                        tx_type = 'DEBIT'
                        debit   = amount_val
                        credit  = None
                        amount  = -amount_val
                    elif sign == '+':
                        tx_type = 'CREDIT'
                        credit  = amount_val
                        debit   = None
                        amount  = amount_val
                    else:
                        # No sign — deduce from description context
                        # CREDIT keywords: VIR, VIREMENT, EVI (employer transfer), GAB DEP, REMISE, INTERETS
                        desc_up = desc.upper()
                        credit_keywords = ['EVI ', 'VIREMENT', 'GAB ', 'REMISE', 'INTERETS',
                                           'AVOIR', 'REMBOURSEMENT', 'CPAM', 'NOVEOCARE',
                                           'MANGOPAY', 'PAYPAL']
                        is_credit = any(kw in desc_up for kw in credit_keywords)
                        if is_credit:
                            tx_type = 'CREDIT'
                            credit  = amount_val
                            debit   = None
                            amount  = amount_val
                        else:
                            tx_type = 'DEBIT'
                            debit   = amount_val
                            credit  = None
                            amount  = -amount_val

                    transactions.append({
                        'date_compta':    full_date(date_c),
                        'date_operation': full_date(date_o),
                        'date_valeur':    full_date(date_v),
                        'description':    desc.strip(),
                        'debit':          debit,
                        'credit':         credit,
                        'amount':         amount,
                        'type':           tx_type,
                        'source_file':    pdf_path.name,
                    })
                    i = j
                else:
                    i += 1
    return transactions

# ── Dispatcher ────────────────────────────────────────────────────────────────

def parse_pdf(pdf_path: Path) -> list[dict]:
    """Detect PDF format and route to the correct parser."""
    year = extract_year_from_filename(pdf_path.name)

    # Detect format by checking first page text
    with pdfplumber.open(pdf_path) as pdf:
        first_text = pdf.pages[0].extract_text() or ''
        has_tables = len(pdf.pages[0].extract_tables()) > 0

    if 'DETAIL DES OPERATIONS' in first_text:
        return parse_text_format(pdf_path, year)
    elif has_tables:
        return parse_table_format(pdf_path, year)
    else:
        return []

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_EXCEL.parent.mkdir(exist_ok=True)

    pdf_files = sorted(STATEMENTS_DIR.glob("*.pdf"))
    log.info("Found %d PDF statements", len(pdf_files))
    print(f"Found {len(pdf_files)} PDF statements\n")

    all_transactions = []

    for pdf_path in pdf_files:
        print(f"Parsing: {pdf_path.name[:60]}...")
        try:
            txs = parse_pdf(pdf_path)
            log.info("Parsed %s -> %d transactions", pdf_path.name, len(txs))
            print(f"  -> {len(txs)} transactions extracted")
            all_transactions.extend(txs)
        except Exception as e:
            log.error("Failed to parse %s: %s", pdf_path.name, e)
            print(f"  ERROR: {e}")

    if not all_transactions:
        log.error("No transactions extracted from any PDF. Check PDF structure.")
        print("\nNo transactions extracted. Check PDF structure.")
        return

    df = pd.DataFrame(all_transactions)

    # Convert dates
    for col in ['date_compta', 'date_operation', 'date_valeur']:
        df[col] = pd.to_datetime(df[col], format='%d/%m/%Y', errors='coerce')

    # Sort chronologically
    df = df.sort_values('date_operation').reset_index(drop=True)

    # ── Validate parsed output ─────────────────────────────────────────────────
    from src.schemas import validate_transactions
    df = validate_transactions(df, source="parse_statements")

    # ── Build monthly summary sheet ───────────────────────────────────────────
    df['year_month'] = df['date_operation'].dt.to_period('M').astype(str)
    monthly = df.groupby('year_month').agg(
        total_debit  =('debit',  'sum'),
        total_credit =('credit', 'sum'),
        num_transactions=('amount', 'count'),
        net_flow     =('amount', 'sum'),
    ).reset_index()

    # ── Save to Excel (kept for backwards compat) and SQLite ─────────────────
    with pd.ExcelWriter(OUTPUT_EXCEL, engine='openpyxl') as writer:
        # Sheet 1 — all transactions
        df.drop(columns=['year_month']).to_excel(writer, sheet_name='Transactions', index=False)

        # Sheet 2 — monthly summary
        monthly.to_excel(writer, sheet_name='Monthly Summary', index=False)

        # Sheet 3 — stats overview
        stats = pd.DataFrame({
            'Metric': [
                'Total Transactions',
                'Date From',
                'Date To',
                'Total Debits (€)',
                'Total Credits (€)',
                'Net Flow (€)',
                'Avg Monthly Spend (€)',
                'Avg Monthly Income (€)',
            ],
            'Value': [
                len(df),
                str(df['date_operation'].min().date()),
                str(df['date_operation'].max().date()),
                round(df['debit'].sum(), 2),
                round(df['credit'].sum(), 2),
                round(df['amount'].sum(), 2),
                round(monthly['total_debit'].mean(), 2),
                round(monthly['total_credit'].mean(), 2),
            ]
        })
        stats.to_excel(writer, sheet_name='Overview', index=False)

    # ── Write to SQLite ───────────────────────────────────────────────────────
    write_table(df.drop(columns=['year_month']), "transactions")
    write_table(monthly, "tx_monthly_summary")
    log.info("Saved %d transactions -> %s + SQLite", len(df), OUTPUT_EXCEL)

    # Summary
    print(f"\n{'='*55}")
    print(f"EXTRACTION COMPLETE")
    print(f"{'='*55}")
    print(f"Total transactions : {len(df)}")
    print(f"Date range         : {df['date_operation'].min().date()} to {df['date_operation'].max().date()}")
    print(f"Total debits       : {df['debit'].sum():,.2f} €")
    print(f"Total credits      : {df['credit'].sum():,.2f} €")
    print(f"Output saved to    : {OUTPUT_EXCEL}")
    print(f"\nSheets: Transactions | Monthly Summary | Overview")
    print(f"\nFirst 5 rows:")
    print(df[['date_operation', 'description', 'amount', 'type']].head().to_string(index=False))

if __name__ == "__main__":
    main()
