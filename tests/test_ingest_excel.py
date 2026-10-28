"""Workbook-wide allocations must be bounded before openpyxl opens XLSX."""

import importlib
import struct
import zipfile
from xml.etree import ElementTree

import numpy as np
import pytest

pytest.importorskip("pyarrow")
openpyxl = pytest.importorskip("openpyxl")

from pybiglasso.ingest import ingest  # noqa: E402


@pytest.fixture
def shared_strings_workbook(tmp_path):
    # Independent XLSX source: openpyxl writes inline strings; transform the
    # unselected cells into a real workbook-wide shared-string table.
    source = tmp_path / "shared.xlsx"
    base = tmp_path / "base.xlsx"
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.append(["x", "ignored"])
    for index in range(4096):
        sheet.append([index, None])
    workbook.save(base)
    workbook.close()
    ns = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    content_ns = "http://schemas.openxmlformats.org/package/2006/content-types"
    rel_ns = "http://schemas.openxmlformats.org/package/2006/relationships"
    with zipfile.ZipFile(base) as archive:
        files = {name: archive.read(name) for name in archive.namelist()}
    root = ElementTree.fromstring(files["xl/worksheets/sheet1.xml"])
    for index, row in enumerate(root.find(f"{{{ns}}}sheetData")[1:]):
        cell = ElementTree.SubElement(row, f"{{{ns}}}c", r=f"B{index + 2}", t="s")
        ElementTree.SubElement(cell, f"{{{ns}}}v").text = str(index)
    files["xl/worksheets/sheet1.xml"] = ElementTree.tostring(root)
    strings = ElementTree.Element(f"{{{ns}}}sst", count="4096", uniqueCount="4096")
    for index in range(4096):
        item = ElementTree.SubElement(strings, f"{{{ns}}}si")
        ElementTree.SubElement(item, f"{{{ns}}}t").text = f"unique-{index}-" + "z" * 512
    files["xl/sharedStrings.xml"] = ElementTree.tostring(strings)
    content = ElementTree.fromstring(files["[Content_Types].xml"])
    ElementTree.SubElement(
        content,
        f"{{{content_ns}}}Override",
        PartName="/xl/sharedStrings.xml",
        ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sharedStrings+xml",
    )
    files["[Content_Types].xml"] = ElementTree.tostring(content)
    rels = ElementTree.fromstring(files["xl/_rels/workbook.xml.rels"])
    ElementTree.SubElement(
        rels,
        f"{{{rel_ns}}}Relationship",
        Id="rIdShared",
        Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/sharedStrings",
        Target="sharedStrings.xml",
    )
    files["xl/_rels/workbook.xml.rels"] = ElementTree.tostring(rels)
    with zipfile.ZipFile(source, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in files.items():
            archive.writestr(name, data)
    return source


def test_large_unselected_shared_strings_rejected_before_reader(
    shared_strings_workbook, tmp_path, monkeypatch
):
    def forbidden(*args, **kwargs):
        pytest.fail("openpyxl opened an over-budget workbook")

    monkeypatch.setattr(openpyxl, "load_workbook", forbidden)
    with pytest.raises(ValueError, match="metadata exceeds excel_metadata_limit"):
        ingest(
            shared_strings_workbook, tmp_path / "X.arrow", features=["x"], block_size=1
        )
    assert not (tmp_path / "X.arrow").exists()


def test_explicit_larger_budget_has_valid_source_positive_twin(
    shared_strings_workbook, tmp_path, monkeypatch
):
    original = openpyxl.load_workbook
    independent = original(shared_strings_workbook, read_only=True, data_only=True)
    try:
        rows = list(independent.active.values)
    finally:
        independent.close()
    assert len(rows) == 4097
    assert rows[-1][1] == "unique-4095-" + "z" * 512
    loads = []

    def counted(*args, **kwargs):
        workbook = original(*args, **kwargs)
        loads.append(len(workbook.active._shared_strings))
        return workbook

    monkeypatch.setattr(openpyxl, "load_workbook", counted)
    result = ingest(
        shared_strings_workbook,
        tmp_path / "X.arrow",
        features=["x"],
        block_size=1,
        excel_metadata_limit=8 << 20,
    )
    assert loads == [4096, 4096, 4096]
    np.testing.assert_array_equal(
        result.design._columns[0][0], [row[0] for row in rows[1:]]
    )


def test_preflight_is_checked_again_before_each_pass(
    shared_strings_workbook, tmp_path, monkeypatch
):
    module = importlib.import_module("pybiglasso.ingest")
    original = module.check_metadata
    checks = []

    def counted(source, limit):
        checks.append(limit)
        return original(source, limit)

    monkeypatch.setattr(module, "check_metadata", counted)
    ingest(
        shared_strings_workbook,
        tmp_path / "X.arrow",
        features=["x"],
        excel_metadata_limit=8 << 20,
    )
    assert checks == [8 << 20] * 3


def test_zip_directory_rejected_before_zipfile_allocates(tmp_path, monkeypatch):
    source = tmp_path / "many-parts.xlsx"
    with zipfile.ZipFile(source, "w") as archive:
        for index in range(1025):
            archive.writestr(f"part-{index}", b"")
    module = importlib.import_module("pybiglasso._xlsx")
    monkeypatch.setattr(
        module.zipfile, "ZipFile", lambda *_: pytest.fail("Directory allocated")
    )
    with pytest.raises(ValueError, match="directory exceeds"):
        ingest(source, tmp_path / "X.arrow")


def test_misleading_directory_signature_in_comment_rejected(tmp_path, monkeypatch):
    source = tmp_path / "comment.xlsx"
    with zipfile.ZipFile(source, "w") as archive:
        archive.writestr("part", b"")
        archive.comment = struct.pack(
            "<4s4H2LH", b"PK\x05\x06", 0, 0, 1, 1, 1 << 30, 0, 65535
        )
    module = importlib.import_module("pybiglasso._xlsx")
    monkeypatch.setattr(
        module.zipfile, "ZipFile", lambda *_: pytest.fail("Directory allocated")
    )
    with pytest.raises(ValueError, match="Invalid XLSX ZIP directory"):
        ingest(source, tmp_path / "X.arrow")


def test_styles_are_counted_even_without_shared_strings(tmp_path, monkeypatch):
    base = tmp_path / "base.xlsx"
    source = tmp_path / "styles.xlsx"
    workbook = openpyxl.Workbook()
    workbook.active.append(["x"])
    workbook.active.append([1])
    workbook.save(base)
    workbook.close()
    with (
        zipfile.ZipFile(base) as original,
        zipfile.ZipFile(source, "w", compression=zipfile.ZIP_DEFLATED) as output,
    ):
        for name in original.namelist():
            data = original.read(name)
            if name == "xl/styles.xml":
                data += b" " * (1 << 20)
            output.writestr(name, data)
    monkeypatch.setattr(
        openpyxl, "load_workbook", lambda *_, **__: pytest.fail("Workbook opened")
    )
    with pytest.raises(ValueError, match="metadata exceeds excel_metadata_limit"):
        ingest(source, tmp_path / "X.arrow")


def test_zip_directory_byte_budget_precedes_zipfile(tmp_path, monkeypatch):
    source = tmp_path / "directory.xlsx"
    with zipfile.ZipFile(source, "w") as archive:
        for index in range(1024):
            archive.writestr(str(index) + "x" * 1100, b"")
    module = importlib.import_module("pybiglasso._xlsx")
    monkeypatch.setattr(
        module.zipfile, "ZipFile", lambda *_: pytest.fail("Directory allocated")
    )
    with pytest.raises(ValueError, match="directory exceeds"):
        ingest(source, tmp_path / "X.arrow")


@pytest.mark.parametrize("limit", [0, -1, True, 1.5])
def test_metadata_limit_must_be_positive_integer(tmp_path, limit):
    source = tmp_path / "source.csv"
    source.write_text("x\n1\n2\n")
    with pytest.raises(
        ValueError, match="excel_metadata_limit must be a positive integer"
    ):
        ingest(source, tmp_path / "X.arrow", excel_metadata_limit=limit)
