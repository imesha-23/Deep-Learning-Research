import pandas as pd
import re

DATE_RE = re.compile(r"^\d{1,2}/\d{1,2}/\d{4}")


def read_clean(path):
    df = pd.read_csv(path)
    # Some rows are missing source_type, so every field after it shifts one
    # column left: the datestamp lands in source_type and the content lands in
    # datestamp. Detect those rows and restore the alignment, leaving
    # source_type empty.
    misaligned = df["source_type"].astype(str).str.match(DATE_RE, na=False)
    df.loc[misaligned, "content"] = df.loc[misaligned, "datestamp"]
    df.loc[misaligned, "datestamp"] = df.loc[misaligned, "source_type"]
    df.loc[misaligned, "source_type"] = ""
    df["source_type"] = df["source_type"].fillna("").str.strip()
    return df


# Read CSV files
true_df = read_clean("TRUE.csv")
false_df = read_clean("FALSE.csv")

# Random sample 110 from each
true_sample = true_df.sample(n=171, random_state=42)
false_sample = false_df.sample(n=155, random_state=42)

# Add annotater column
true_sample["annotater"] = "true"
false_sample["annotater"] = "false"

# Combine, shuffle, and drop id column
combined = pd.concat([true_sample, false_sample], ignore_index=True)
combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)
combined = combined.drop(columns=["id"])

# Save to CSV
combined.to_csv("annotated_data.csv", index=False)

print(f"Created annotated_data.csv with {len(combined)} rows ({len(true_sample)} true + {len(false_sample)} false)")
