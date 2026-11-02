"""Excel memory exception and directory-only shared-string size guard."""

import copy
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


def test_over_ceiling_rejected_without_decompression(
    shared_strings_workbook, tmp_path, monkeypatch
):
    with zipfile.ZipFile(shared_strings_workbook) as archive:
        size = archive.getinfo("xl/sharedStrings.xml").file_size

    def forbidden(*args, **kwargs):
        pytest.fail("Over-ceiling shared strings were materialized")

    monkeypatch.setattr(openpyxl, "load_workbook", forbidden)
    monkeypatch.setattr(zipfile.ZipFile, "open", forbidden)
    with pytest.raises(ValueError, match="convert the workbook to CSV or Parquet"):
        ingest(
            shared_strings_workbook,
            tmp_path / "X.arrow",
            features=["x"],
            block_size=1,
            excel_shared_strings_limit=size - 1,
        )
    assert not (tmp_path / "X.arrow").exists()


@pytest.mark.parametrize("over_ceiling", [False, True])
def test_renamed_shared_strings_part(
    shared_strings_workbook, tmp_path, monkeypatch, over_ceiling
):
    with zipfile.ZipFile(shared_strings_workbook) as archive:
        files = {name: archive.read(name) for name in archive.namelist()}
    files["xl/strings.xml"] = files.pop("xl/sharedStrings.xml")
    for name in ("[Content_Types].xml", "xl/_rels/workbook.xml.rels"):
        files[name] = files[name].replace(b"sharedStrings.xml", b"strings.xml")
    source = tmp_path / "renamed.xlsx"
    with zipfile.ZipFile(source, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in files.items():
            archive.writestr(name, data)
    output = tmp_path / "renamed.arrow"
    if over_ceiling:
        original_open = zipfile.ZipFile.open

        def checked_open(archive, name, *args, **kwargs):
            assert name != "xl/strings.xml", "Shared strings decompressed"
            return original_open(archive, name, *args, **kwargs)

        monkeypatch.setattr(zipfile.ZipFile, "open", checked_open)
        monkeypatch.setattr(
            openpyxl, "load_workbook", lambda *_, **__: pytest.fail("Workbook opened")
        )
        with pytest.raises(ValueError, match="exceeds excel_shared_strings_limit"):
            ingest(source, output, features=["x"], excel_shared_strings_limit=1)
        assert not output.exists()
    else:
        result = ingest(
            source,
            output,
            features=["x"],
            excel_shared_strings_limit=len(files["xl/strings.xml"]),
        )
        np.testing.assert_array_equal(result.X.column(0).to_numpy(), np.arange(4096))


@pytest.mark.parametrize("explicit_ceiling", [False, True])
def test_under_ceiling_ingests_with_workbook_memory_exception(
    shared_strings_workbook, tmp_path, monkeypatch, explicit_ceiling
):
    original = openpyxl.load_workbook
    independent = original(shared_strings_workbook, read_only=True, data_only=True)
    try:
        rows = list(independent.active.values)
    finally:
        independent.close()
    assert len(rows) == 4097
    assert rows[-1][1] == "unique-4095-" + "z" * 512
    with zipfile.ZipFile(shared_strings_workbook) as archive:
        size = archive.getinfo("xl/sharedStrings.xml").file_size
    loads = []

    def counted(*args, **kwargs):
        workbook = original(*args, **kwargs)
        loads.append(len(workbook.active._shared_strings))
        return workbook

    monkeypatch.setattr(openpyxl, "load_workbook", counted)
    options = {"excel_shared_strings_limit": size} if explicit_ceiling else {}
    result = ingest(
        shared_strings_workbook,
        tmp_path / "X.arrow",
        features=["x"],
        block_size=1,
        **options,
    )
    # This is an intentional format exception, not a block-bounded claim.
    assert loads == [4096, 4096, 4096]
    np.testing.assert_array_equal(
        result.X.column(0).to_numpy(), [row[0] for row in rows[1:]]
    )


def test_default_one_gib_ceiling_uses_directory_sizes(
    shared_strings_workbook, tmp_path, monkeypatch
):
    original = zipfile.ZipFile.infolist

    def declared_large(archive):
        members = [copy.copy(info) for info in original(archive)]
        for member in members:
            if member.filename == "xl/sharedStrings.xml":
                member.file_size = (1 << 30) + 1
        return members

    monkeypatch.setattr(zipfile.ZipFile, "infolist", declared_large)
    monkeypatch.setattr(
        openpyxl, "load_workbook", lambda *_, **__: pytest.fail("Workbook opened")
    )
    monkeypatch.setattr(
        zipfile.ZipFile, "open", lambda *_, **__: pytest.fail("Member decompressed")
    )
    with pytest.raises(ValueError, match="exceeds excel_shared_strings_limit"):
        ingest(shared_strings_workbook, tmp_path / "X.arrow", features=["x"])


def test_explicit_ceiling_can_be_raised(shared_strings_workbook, monkeypatch):
    module = importlib.import_module("pybiglasso.ingest")
    original = zipfile.ZipFile.infolist

    def declared_large(archive):
        members = [copy.copy(info) for info in original(archive)]
        for member in members:
            if member.filename == "xl/sharedStrings.xml":
                member.file_size = (1 << 30) + 1
        return members

    monkeypatch.setattr(zipfile.ZipFile, "infolist", declared_large)
    original_open = zipfile.ZipFile.open

    def manifest_only(archive, member, *args, **kwargs):
        assert member.filename == "[Content_Types].xml"
        return original_open(archive, member, *args, **kwargs)

    monkeypatch.setattr(zipfile.ZipFile, "open", manifest_only)
    module._check_shared_strings(shared_strings_workbook, (1 << 30) + 1)


@pytest.mark.parametrize("case", ["oversized", "malformed", "dtd", "missing part"])
def test_unverifiable_manifest_is_rejected_before_workbook_open(
    tmp_path, monkeypatch, case
):
    source = tmp_path / "manifest.xlsx"
    manifest = {
        "oversized": b" " * ((1 << 20) + 1),
        "malformed": b"<Types>",
        "dtd": b'<!DOCTYPE Types [<!ENTITY part "strings">]><Types/>',
        "missing part": (
            b'<Types><Override PartName="/xl/missing.xml" ContentType="application/'
            b'vnd.openxmlformats-officedocument.spreadsheetml.sharedStrings+xml"/></Types>'
        ),
    }[case]
    with zipfile.ZipFile(source, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", manifest)
    monkeypatch.setattr(
        openpyxl, "load_workbook", lambda *_, **__: pytest.fail("Workbook opened")
    )
    if case == "oversized":
        monkeypatch.setattr(
            zipfile.ZipFile, "open", lambda *_, **__: pytest.fail("Manifest decoded")
        )
    with pytest.raises(ValueError, match="XLSX"):
        ingest(source, tmp_path / "X.arrow", features=["x"])


def test_all_declared_shared_string_parts_count_toward_ceiling(tmp_path):
    module = importlib.import_module("pybiglasso.ingest")
    source = tmp_path / "multiple.xlsx"
    manifest = ElementTree.Element("Types")
    with zipfile.ZipFile(source, "w") as archive:
        for name in ("xl/first.xml", "xl/second.xml"):
            ElementTree.SubElement(
                manifest,
                "Override",
                PartName="/" + name,
                ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sharedStrings+xml",
            )
            archive.writestr(name, b"12345")
        archive.writestr("[Content_Types].xml", ElementTree.tostring(manifest))
    with pytest.raises(ValueError, match="exceeds excel_shared_strings_limit"):
        module._check_shared_strings(source, 9)
    module._check_shared_strings(source, 10)


def test_preflight_is_checked_before_each_pass(
    shared_strings_workbook, tmp_path, monkeypatch
):
    module = importlib.import_module("pybiglasso.ingest")
    original = module._check_shared_strings
    checks = []

    def counted(source, limit):
        checks.append(limit)
        return original(source, limit)

    monkeypatch.setattr(module, "_check_shared_strings", counted)
    ingest(shared_strings_workbook, tmp_path / "X.arrow", features=["x"])
    assert checks == [1 << 30] * 3


@pytest.mark.parametrize("limit", [0, -1, True, 1.5])
def test_ceiling_must_be_positive_integer(tmp_path, limit):
    source = tmp_path / "source.csv"
    source.write_text("x\n1\n2\n")
    with pytest.raises(
        ValueError, match="excel_shared_strings_limit must be a positive integer"
    ):
        ingest(source, tmp_path / "X.arrow", excel_shared_strings_limit=limit)


@pytest.mark.parametrize("case", ["entries", "bytes", "comment"])
def test_directory_caps_precede_zipfile_allocation(tmp_path, monkeypatch, case):
    source = tmp_path / "directory.xlsx"
    with zipfile.ZipFile(source, "w") as archive:
        if case == "entries":
            for index in range(1025):
                archive.writestr(f"part-{index}", b"")
        elif case == "bytes":
            for index in range(1024):
                archive.writestr(str(index) + "x" * 1100, b"")
        else:
            archive.writestr("part", b"")
            archive.comment = struct.pack(
                "<4s4H2LH", b"PK\x05\x06", 0, 0, 1, 1, 1 << 30, 0, 65535
            )
    monkeypatch.setattr(
        zipfile, "ZipFile", lambda *_: pytest.fail("Directory allocated")
    )
    with pytest.raises(ValueError, match="XLSX ZIP directory"):
        ingest(source, tmp_path / "X.arrow")
