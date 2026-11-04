"""Developer probe of a malformed Arrow producer, not a behavioral regression test.

Run with: uv run python scripts/probe_arrow_stream_schema.py

The C stream has one declared schema. PyArrow 26 imports the mislabeled int16
buffers below using int8, silently changing values; full validation passes.
Future Arrow releases may reject this producer instead. Neither outcome is
required here. Producers must compare their batches with the intended schema
BEFORE exporting, while their original types are still observable.
"""

import json

import pyarrow as pa


def main():
    declared = pa.schema([("x", pa.int8())])
    batch = pa.record_batch(
        [pa.array([1, 256, -2, 1024], type=pa.int16())], names=["x"]
    )
    report = {
        "pyarrow": pa.__version__,
        "declared_schema": str(declared),
        "original_schema": str(batch.schema),
        "original_values": batch.column(0).to_pylist(),
        "matches_intended_before_export": batch.schema.equals(
            declared, check_metadata=True
        ),
    }
    try:
        # Deliberately violate the producer contract. The original buffer is
        # longer than the declared type needs; this does not probe short buffers.
        source = pa.RecordBatchReader.from_batches(declared, [batch])
        with pa.RecordBatchReader.from_stream(source) as reader:
            imported = next(reader)
            report["imported_schema"] = str(imported.schema)
            report["imported_values"] = imported.column(0).to_pylist()
            imported.validate(full=True)
            report["full_validation"] = "passed"
    except Exception as error:
        report["rejected"] = f"{type(error).__name__}: {error}"
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
