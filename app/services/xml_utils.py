from __future__ import annotations

import xml.etree.ElementTree as ET

from memu.utils.xml import XML_TOKEN, escape_xml_prose

_PROSE_TAGS = {"narrative_self", "description", "prose", "section", "body",
               "add", "remove", "intention", "companion_memory"}


def extract_xml_fragment(raw: str, root_tag: str) -> ET.Element:
    text = str(raw or "").strip()
    start, depth = None, 0
    for match in XML_TOKEN.finditer(text):
        if match.lastgroup != "tag":
            continue
        token = match.group(0)
        name = token.lstrip("</").split(None, 1)[0].rstrip("/>")
        if name != root_tag or (start is None and token.startswith("</")):
            continue
        if start is None:
            start = match.start()
        depth += -1 if token.startswith("</") else 0 if token.endswith("/>") else 1
        if depth == 0:
            fragment = text[start:match.end()]
            break
    else:
        raise ValueError(f"Missing <{root_tag}> root")
    return ET.fromstring(escape_xml_prose(fragment, _PROSE_TAGS))


def xml_text(element: ET.Element | None, path: str) -> str | None:
    if element is None:
        return None
    node = element.find(path)
    if node is None:
        return None
    if node.attrib or list(node):
        raise ValueError(f"Expected plain XML text for {path}")
    if node.text is None:
        return None
    text = str(node.text).strip()
    return text or None
