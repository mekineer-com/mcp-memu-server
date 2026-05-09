from __future__ import annotations

import re
import xml.etree.ElementTree as ET


def extract_xml_fragment(raw: str, root_tag: str) -> ET.Element:
    text = str(raw or "").strip()
    match = re.search(rf"<{root_tag}(?:\s[^>]*)?>(.*?)</{root_tag}>", text, re.DOTALL)
    if match is None:
        raise ValueError(f"Missing <{root_tag}> root")
    return ET.fromstring(f"<{root_tag}>{match.group(1)}</{root_tag}>")


def xml_text(element: ET.Element | None, path: str) -> str | None:
    if element is None:
        return None
    node = element.find(path)
    if node is None or node.text is None:
        return None
    text = str(node.text).strip()
    return text or None
