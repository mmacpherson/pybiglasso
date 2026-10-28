"""Bound XLSX metadata before a reader can eagerly decode shared strings."""

import struct
import zipfile
from xml.etree import ElementTree

_DIRECTORY_BYTES = 1 << 20
_DIRECTORY_ENTRIES = 1024
_WORKSHEET_TYPE = (
    "application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"
)


def check_metadata(source, limit):
    # ZipFile itself reads the complete central directory. Bound that before
    # constructing it, rather than inspecting infolist after allocation.
    with source.open("rb") as stream:
        stream.seek(0, 2)
        size = stream.tell()
        stream.seek(max(0, size - (65535 + 22 + 20)))
        tail = stream.read()
    # Match ZipFile's last-signature choice. Never validate an earlier EOCD
    # while ZipFile could instead read a forged large directory in a comment.
    offset = tail.rfind(b"PK\x05\x06")
    if offset < 0 or offset + 22 > len(tail):
        raise ValueError("Invalid XLSX ZIP directory.")
    fields = struct.unpack_from("<4s4H2LH", tail, offset)
    if offset + 22 + fields[-1] != len(tail):
        raise ValueError("Invalid XLSX ZIP directory.")
    _, disk, start_disk, disk_entries, entries, directory_size, _, _ = fields
    if (
        disk
        or start_disk
        or disk_entries != entries
        or entries > _DIRECTORY_ENTRIES
        or directory_size > _DIRECTORY_BYTES
        or tail[max(0, offset - 20) : offset - 16] == b"PK\x06\x07"
    ):
        raise ValueError("XLSX directory exceeds the bounded v1 archive contract.")

    with zipfile.ZipFile(source) as archive:
        members = archive.infolist()
        if (
            len(members) != entries
            or len({item.filename for item in members}) != entries
        ):
            raise ValueError("Invalid or duplicate XLSX archive members.")
        content = archive.getinfo("[Content_Types].xml")
        if content.file_size > limit:
            raise ValueError("XLSX metadata exceeds excel_metadata_limit.")
        manifest = archive.read(content)
        _check_xml(manifest)
        tree = ElementTree.fromstring(manifest)
        parts = {}
        for element in tree:
            if element.tag.rsplit("}", 1)[-1] == "Override":
                name = element.attrib["PartName"].removeprefix("/")
                if name in parts:
                    raise ValueError("Duplicate XLSX content-type override.")
                parts[name] = element.attrib["ContentType"]
        # Only standard worksheet XML is lazy. Shared strings, styles, theme,
        # relationships and every other member count against the budget,
        # including metadata for unselected sheets/columns.
        worksheets = {
            name
            for name, kind in parts.items()
            if kind == _WORKSHEET_TYPE
            and name.startswith("xl/worksheets/")
            and name.endswith(".xml")
        }
        metadata = [item for item in members if item.filename not in worksheets]
        if sum(item.file_size for item in metadata) > limit:
            raise ValueError("XLSX metadata exceeds excel_metadata_limit.")
        # V1 rejects DTD/entity declarations and NUL-bearing XML. Scan in
        # bounded chunks before openpyxl parses any of the eager metadata.
        for item in metadata:
            if not item.filename.endswith((".xml", ".rels")):
                continue
            with archive.open(item) as stream:
                carry = b""
                while chunk := stream.read(65536):
                    _check_xml(carry + chunk)
                    carry = chunk[-16:]


def _check_xml(data):
    if b"\x00" in data or b"<!DOCTYPE" in data or b"<!ENTITY" in data:
        raise ValueError("XLSX metadata must not contain NUL bytes or DTDs/entities.")
