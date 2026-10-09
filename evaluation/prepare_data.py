"""Download the Zenodo PPI dataset and derive the evaluation variants + ground truth.

The outputs are committed; rerun only to reproduce or verify them.

Usage: uv run python -m evaluation.prepare_data [out_dir]   (default: evaluation/data)

Dataset: Carreño, E. (2026). Synthetic Protein Interaction Dataset (1.0.5).
Zenodo. https://doi.org/10.5281/zenodo.21455347 (MIT license)
"""

import csv
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

RECORD = "https://zenodo.org/api/records/21455347/files"
FILES = {
    "synthetic_protein_interactions.tsv": "155577b25e2e8460a59e8a096875edfa",
    "croissant.jsonld": "0ad5e18be753bc06b7993fed50f6ccb1",
}
TSV = "synthetic_protein_interactions.tsv"
FLAGS = (
    "is_directed",
    "is_stimulation",
    "is_inhibition",
    "consensus_direction",
    "consensus_stimulation",
    "consensus_inhibition",
)


def download(out: Path) -> None:
    for name, md5 in FILES.items():
        target = out / "raw" / name
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(f"{RECORD}/{name}/content", target)
        digest = hashlib.md5(target.read_bytes()).hexdigest()
        if digest != md5:
            sys.exit(f"checksum mismatch for {name}: {digest} != {md5}")


def normalize(row: dict) -> dict:
    row = {k: v.strip() for k, v in row.items()}
    row["source_genesymbol"] = row["source_genesymbol"].upper()
    row["target_genesymbol"] = row["target_genesymbol"].upper()
    return row


def main(out: Path) -> None:
    download(out)
    with open(out / "raw" / TSV, newline="") as fh:
        raw = list(csv.DictReader(fh, delimiter="\t"))
    fieldnames = list(raw[0])

    # Deduplicate after symbol normalization: this removes the documented
    # exact duplicate and the MYC->GAPDH/gapdh duplicate hidden by case.
    unique, seen = [], set()
    for row in map(normalize, raw):
        key = tuple(row.values())
        if key not in seen:
            seen.add(key)
            unique.append(row)

    # Clean variant (L3/L4): same columns, issues resolved, original row order.
    clean = out / "clean"
    clean.mkdir(parents=True, exist_ok=True)
    with open(clean / TSV, "w", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=fieldnames, delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(unique)

    # Original variant (L5/L6): the file exactly as published.
    original = out / "original"
    original.mkdir(parents=True, exist_ok=True)
    (original / TSV).write_bytes((out / "raw" / TSV).read_bytes())

    nodes = {}
    for row in unique:
        for side in ("source", "target"):
            nodes[row[side]] = {
                "id": row[side],
                "genesymbol": row[f"{side}_genesymbol"],
                "ncbi_tax_id": int(row[f"ncbi_tax_id_{side}"]),
            }
    edges = [
        {
            "id": f"{row['source']}_{row['target']}_{row['type']}",
            "source": row["source"],
            "target": row["target"],
            "type": row["type"],
            **{flag: row[flag] == "1" for flag in FLAGS},
        }
        for row in unique
    ]
    truth = {
        "dataset_doi": "10.5281/zenodo.21455347",
        "raw_rows": len(raw),
        "nodes": sorted(nodes.values(), key=lambda n: n["id"]),
        "edges": edges,
    }
    (out / "ground_truth.json").write_text(json.dumps(truth, indent=2) + "\n")
    print(f"{len(raw)} raw rows -> {len(nodes)} nodes, {len(edges)} edges")


if __name__ == "__main__":
    from evaluation.config import DATA_DIR

    main(Path(sys.argv[1]) if len(sys.argv) > 1 else DATA_DIR)
