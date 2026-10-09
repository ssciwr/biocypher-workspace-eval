"""Inspect a graded project from inside its own environment.

Runs as ``uv run --project <project> python probe.py <mode> ...`` so the
model's adapter is imported with the dependencies the model declared, not
with the registry's. Standard library only (pyarrow for ``parquet`` mode,
which only makes sense when the graph build, which needs pyarrow, succeeded).
Prints one JSON object on a line prefixed with MARKER (the graded code
may print to stdout too).

Modes:
    adapter <adapter_file> <data_file> [<module_name>]
        Instantiate SyntheticPpiAdapter(data_source=<data_file>) and dump
        node and edge tuples with property types, iterating twice.
    parquet <output_dir>
        Row counts and column types of the BioCypher Parquet output.
"""

import importlib
import importlib.util
import json
import sys
import traceback
from pathlib import Path

CLASS_NAME = "SyntheticPpiAdapter"
MARKER = "@@PROBE@@"


def _typed(props) -> dict:
    """Properties as {key: [json-safe value, type name]}."""
    if not isinstance(props, dict):
        return {"__not_a_dict__": [repr(props), type(props).__name__]}
    out = {}
    for key, value in props.items():
        safe = (
            value
            if isinstance(value, (str, int, float, bool, type(None)))
            else repr(value)
        )
        out[str(key)] = [safe, type(value).__name__]
    return out


def _tuple(item, props_index: int) -> dict:
    if not isinstance(item, tuple):
        return {"malformed": repr(item)[:200]}
    fields = [repr(x)[:200] if not isinstance(x, str) else x for x in item]
    record = {"length": len(item), "fields": fields}
    if len(item) > props_index:
        record["props"] = _typed(item[props_index])
    return record


def load_class(adapter_file: str, module_name: str | None):
    if module_name:
        return getattr(importlib.import_module(module_name), CLASS_NAME)
    spec = importlib.util.spec_from_file_location("graded_adapter", adapter_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, CLASS_NAME)


def probe_adapter(
    adapter_file: str, data_file: str, module_name: str | None = None
) -> dict:
    result: dict = {"error": None}
    try:
        cls = load_class(adapter_file, module_name)
        adapter = cls(data_source=data_file)
        nodes = [_tuple(n, 2) for n in adapter.get_nodes()]
        edges = [_tuple(e, 4) for e in adapter.get_edges()]
        nodes_again = [_tuple(n, 2) for n in adapter.get_nodes()]
        edges_again = [_tuple(e, 4) for e in adapter.get_edges()]
        result.update(
            nodes=nodes,
            edges=edges,
            reiterable=nodes == nodes_again and edges == edges_again,
        )
    except Exception:  # noqa: BLE001 - any failure of the graded code is a result
        result["error"] = traceback.format_exc(limit=5)[-2000:]
    return result


def probe_parquet(output_dir: str) -> dict:
    import pyarrow.parquet as pq

    tables = {}
    for path in sorted(Path(output_dir).rglob("*.parquet")):
        table = pq.read_table(path)
        tables[path.name] = {
            "rows": table.num_rows,
            "types": {f.name: str(f.type) for f in table.schema},
        }
    return {"files": tables}


def main(argv: list[str]) -> None:
    mode, *args = argv
    if mode == "adapter":
        out = probe_adapter(*args)
    elif mode == "parquet":
        out = probe_parquet(*args)
    else:
        raise SystemExit(f"unknown mode {mode}")
    print(MARKER + json.dumps(out))


if __name__ == "__main__":
    main(sys.argv[1:])
