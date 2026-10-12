from __future__ import annotations

import re
import xml.etree.ElementTree as ET


_START = r"(?:[A-Za-z_:]|[^\x00-\x7f])"
_NAME = _START + r"(?:[A-Za-z0-9_.:-]|[^\x00-\x7f])*"
# Shield opaque XML and tag candidates; ElementTree still validates the grammar.
_XML_TOKEN = re.compile(
    r"(?P<opaque><!\[CDATA\[.*?\]\]>|<!--.*?-->|<\?.*?\?>)"
    rf"""|(?P<tag></?{_START}(?:[^<>"']|"[^"]*"|'[^']*')*>)"""
    rf"|(?P<damaged></?{_NAME}(?=\s+{_NAME}\s*=|[/=>])[^<>]*)"
    rf"|(?P<bare>&(?!{_NAME};|#[^;\s<&]*;)|<)",
    re.DOTALL,
)
_PROSE_TAGS = {"narrative_self", "description", "prose", "section", "body",
               "add", "remove", "intention", "companion_memory"}


def extract_xml_fragment(raw: str, root_tag: str) -> ET.Element:
    text = str(raw or "").strip()
    start, depth = None, 0
    for match in _XML_TOKEN.finditer(text):
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
    prose_depth = 0

    def escape_prose(match: re.Match[str]) -> str:
        nonlocal prose_depth
        token = match.group(0)
        if match.lastgroup == "tag":
            name = token.lstrip("</").split(None, 1)[0].rstrip("/>")
            if name in _PROSE_TAGS:
                prose_depth += -1 if token.startswith("</") else 0 if token.endswith("/>") else 1
        elif match.lastgroup == "bare" and prose_depth > 0:
            if token == "&":
                return "&amp;"
            if fragment[match.end():match.end() + 1] not in {"/", "!", "?"}:
                return "&lt;"
        return token

    return ET.fromstring(_XML_TOKEN.sub(escape_prose, fragment))


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
