#!/usr/bin/env python3
"""
Split a Sinhala news CSV into separate CSV files by the `type` column.

Outputs are created in the same directory as the input file. Each output
file contains the columns: `domain,datestamp,type,content`.

Usage:
  - Edit the `INPUT_CSV` variable below or run with `--input /path/to/file`.
  - Optional: `--outdir /some/dir` to place output files elsewhere.

This script reads CSV with `utf-8-sig` encoding to handle BOM if present.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
from collections import defaultdict
from typing import Dict, TextIO


def sanitize_filename(name: str) -> str:
	if not name:
		return "UNKNOWN"
	name = name.strip()
	# replace problematic characters with underscore
	name = re.sub(r"[^0-9A-Za-z._-]+", "_", name)
	# limit length
	return name[:120] or "UNKNOWN"


def split_by_type(input_csv: str, out_dir: str) -> Dict[str, int]:
	os.makedirs(out_dir, exist_ok=True)

	writers: Dict[str, csv.writer] = {}
	files: Dict[str, TextIO] = {}
	counts: Dict[str, int] = defaultdict(int)

	with open(input_csv, "r", encoding="utf-8-sig", newline="") as fh:
		reader = csv.DictReader(fh)
		# Decide which field names to use (case-insensitive tolerant)
		fieldnames_lower = {k.lower(): k for k in reader.fieldnames or []}

		domain_field = fieldnames_lower.get("domain")
		datestamp_field = fieldnames_lower.get("datestamp")
		type_field = fieldnames_lower.get("type")
		content_field = fieldnames_lower.get("content")

		if not type_field:
			raise SystemExit("Input CSV does not contain a 'type' column.")

		# Process rows and write to per-type files (streaming)
		for row in reader:
			type_val = (row.get(type_field) or "").strip()
			if not type_val:
				type_val = "UNKNOWN"
			out_name = sanitize_filename(type_val) + ".csv"
			out_path = os.path.join(out_dir, out_name)

			if type_val not in writers:
				# open file and write header
				f = open(out_path, "w", encoding="utf-8", newline="")
				files[type_val] = f
				w = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
				writers[type_val] = w
				w.writerow(["domain", "datestamp", "type", "content"])

			domain_val = (row.get(domain_field) if domain_field else None) or ""
			datestamp_val = (row.get(datestamp_field) if datestamp_field else None) or ""
			content_val = (row.get(content_field) if content_field else None) or ""

			writers[type_val].writerow([domain_val, datestamp_val, type_val, content_val])
			counts[type_val] += 1

	# close files
	for f in files.values():
		f.close()

	return counts


def main() -> None:
	default_input = os.path.join(os.path.dirname(__file__), "sinhalanews.csv")

	p = argparse.ArgumentParser(description="Split sinhalanews CSV into per-type files")
	p.add_argument("--input", "-i", default=default_input, help="Path to sinhalanews.csv")
	p.add_argument("--outdir", "-o", default=os.path.dirname(default_input), help="Output directory")
	args = p.parse_args()

	input_csv = os.path.abspath(args.input)
	out_dir = os.path.abspath(args.outdir)

	if not os.path.exists(input_csv):
		raise SystemExit(f"Input file not found: {input_csv}")

	counts = split_by_type(input_csv, out_dir)

	total = sum(counts.values())
	print(f"Wrote {total} rows into {len(counts)} files in {out_dir}")
	for t, c in sorted(counts.items(), key=lambda x: (-x[1], x[0])):
		print(f"  {t}: {c}")


if __name__ == "__main__":
	main()

