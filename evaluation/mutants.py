"""Deliberately broken variants of the reference adapter.

Used twice: to validate the grader (each mutant must fail at least the
checks listed in ``expected_failures``, the reference must pass all), and to measure the
strength of a model's tests (the fraction of mutants its tests catch).

Mutants are text patches on the reference adapter source from the
reference repo. Every patch must match exactly once, so a changed reference
fails loudly instead of silently producing an unmutated "mutant".
"""

from dataclasses import dataclass

EDGE_PROPS = '{"type": row["type"], **{f: row[f] == "1" for f in FLAGS}}'


@dataclass(frozen=True)
class Mutant:
    name: str
    description: str
    patches: tuple[tuple[str, str], ...]
    expected_failures: frozenset[str]


MUTANTS = (
    Mutant(
        "wrong_id_scheme",
        "edge id without the interaction type",
        (
            (
                "f\"{row['source']}_{row['target']}_{row['type']}\"",
                "f\"{row['source']}_{row['target']}\"",
            ),
        ),
        frozenset({"edge_ids_match"}),
    ),
    Mutant(
        "flags_int",
        "0/1 flags left as int",
        (('row[f] == "1" for f in FLAGS', "int(row[f]) for f in FLAGS"),),
        frozenset({"bool_types", "flags_unchanged", "edge_props_match"}),
    ),
    Mutant(
        "no_dedup",
        "duplicate rows not removed",
        (("                if key not in seen:", "                if True:"),),
        frozenset({"dedup_exact", "dedup_after_case", "edge_tuples_unique"}),
    ),
    Mutant(
        "no_upper",
        "gene symbols not upper-cased",
        (
            (
                'row["source_genesymbol"] = row["source_genesymbol"].upper()',
                'row["source_genesymbol"] = row["source_genesymbol"]',
            ),
            (
                'row["target_genesymbol"] = row["target_genesymbol"].upper()',
                'row["target_genesymbol"] = row["target_genesymbol"]',
            ),
        ),
        frozenset({"symbols_upper", "dedup_after_case", "edge_tuples_unique"}),
    ),
    Mutant(
        "reverse_edge_added",
        "reverse edge added for the undirected row",
        (
            (
                f"                {EDGE_PROPS},\n            )\n",
                (
                    f"                {EDGE_PROPS},\n            )\n"
                    '            if row["is_directed"] == "0":\n'
                    "                yield (\n"
                    "                    f\"{row['target']}_{row['source']}_{row['type']}\",\n"
                    '                    row["target"],\n'
                    '                    row["source"],\n'
                    '                    "protein_protein_interaction",\n'
                    f"                    {EDGE_PROPS},\n"
                    "                )\n"
                ),
            ),
        ),
        # the reverse id collides with the existing directed row, so the id
        # set still matches; duplicates and the direction flag give it away
        frozenset({"no_reverse_added", "edge_tuples_unique"}),
    ),
    Mutant(
        "template_edge_order",
        "edge tuple in the scaffold example's order (source, target, label, label, props)",
        (
            (
                (
                    "                f\"{row['source']}_{row['target']}_{row['type']}\",\n"
                    '                row["source"],\n'
                    '                row["target"],\n'
                    '                "protein_protein_interaction",\n'
                ),
                (
                    '                row["source"],\n'
                    '                row["target"],\n'
                    '                "protein_protein_interaction",\n'
                    '                "protein_protein_interaction",\n'
                ),
            ),
        ),
        # index 3 still holds the label, so only ids/endpoints break
        frozenset({"edge_ids_match", "undirected_kept"}),
    ),
    Mutant(
        "single_pass",
        "file handle opened once, generators exhausted after the first pass",
        (
            (
                "        self.config = kwargs\n",
                (
                    "        self.config = kwargs\n"
                    '        self._fh = open(data_source, newline="")  # noqa: SIM115\n'
                ),
            ),
            (
                '        with open(self.data_source, newline="") as fh:\n',
                "        fh = self._fh\n        if True:\n",
            ),
        ),
        frozenset({"reiterable"}),
    ),
)


def apply(mutant: Mutant, source: str) -> str:
    for old, new in mutant.patches:
        count = source.count(old)
        if count != 1:
            raise ValueError(
                f"mutant {mutant.name}: patch target found {count} times in the "
                "reference adapter; update evaluation/mutants.py"
            )
        source = source.replace(old, new)
    return source
