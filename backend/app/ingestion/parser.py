"""AST-aware code parser using tree-sitter.

Extracts functions, classes, methods, imports, and call sites from source files.
This is the foundation of the chunking strategy — each AST node becomes a
semantically meaningful chunk rather than an arbitrary text split.
"""
from __future__ import annotations
from pathlib import Path

from app.utils.logger import get_logger

log = get_logger(__name__)

# Tree-sitter node types that represent extractable code units
FUNCTION_TYPES = {
    "function_definition",     # Python
    "function_declaration",    # JS/TS/Go/C
    "method_definition",       # Ruby
    "method_declaration",      # Java/C#
    "arrow_function",          # JS/TS (when assigned)
}

CLASS_TYPES = {
    "class_definition",        # Python
    "class_declaration",       # JS/TS/Java/C#
    "struct_declaration",      # Go/Rust
    "impl_item",               # Rust
}

IMPORT_TYPES = {
    "import_statement",        # Python
    "import_from_statement",   # Python
    "import_declaration",      # Java/Go
    "import_specifier",        # JS/TS
}


def _try_import_tree_sitter():
    """Lazy import to avoid startup cost if not used."""
    try:
        import tree_sitter_languages
        return tree_sitter_languages
    except ImportError:
        log.warning("tree-sitter-languages not installed, falling back to regex parser")
        return None


def parse_file_ast(file_path: Path, source_code: str, language: str) -> dict:
    """Parse a source file into structured AST components.

    Returns dict with keys: functions, classes, imports, module_docstring
    Each function/class entry includes: name, content, start_line, end_line,
    docstring, signature, calls, children (for classes).
    """
    ts = _try_import_tree_sitter()
    if ts is None or not language:
        return _fallback_parse(file_path, source_code, language)

    try:
        parser = ts.get_parser(language)
        tree = parser.parse(source_code.encode("utf-8"))
    except Exception as e:
        log.warning(f"tree-sitter parse failed for {file_path}: {e}")
        return _fallback_parse(file_path, source_code, language)

    root = tree.root_node
    result = {
        "functions": [],
        "classes": [],
        "imports": [],
        "module_docstring": _extract_module_docstring(root, source_code, language),
    }

    for node in _walk_top_level(root):
        if node.type in CLASS_TYPES:
            cls_info = _extract_class(node, source_code, language)
            result["classes"].append(cls_info)
        elif node.type in FUNCTION_TYPES:
            fn_info = _extract_function(node, source_code, language)
            result["functions"].append(fn_info)
        elif node.type in IMPORT_TYPES:
            result["imports"].append(source_code[node.start_byte:node.end_byte])

    return result


def _walk_top_level(node):
    """Yield top-level children of the root node."""
    for child in node.children:
        yield child


def _extract_function(node, source_code: str, language: str) -> dict:
    """Extract function info from an AST node."""
    name = _get_name(node)
    content = source_code[node.start_byte:node.end_byte]
    return {
        "name": name,
        "content": content,
        "start_line": node.start_point[0] + 1,
        "end_line": node.end_point[0] + 1,
        "docstring": _extract_docstring(node, source_code, language),
        "signature": _extract_signature(content, language),
        "calls": _extract_calls(node, source_code),
        "parent_class": None,
    }


def _extract_class(node, source_code: str, language: str) -> dict:
    """Extract class info including its methods."""
    name = _get_name(node)
    content = source_code[node.start_byte:node.end_byte]
    methods = []

    for child in node.children:
        # Look for the body node
        if child.type in ("block", "class_body", "declaration_list"):
            for member in child.children:
                if member.type in FUNCTION_TYPES:
                    m = _extract_function(member, source_code, language)
                    m["parent_class"] = name
                    methods.append(m)

    return {
        "name": name,
        "content": content,
        "start_line": node.start_point[0] + 1,
        "end_line": node.end_point[0] + 1,
        "docstring": _extract_docstring(node, source_code, language),
        "signature": f"class {name}",
        "calls": _extract_calls(node, source_code),
        "methods": methods,
    }


def _get_name(node) -> str:
    """Get the name identifier from a function/class node."""
    for child in node.children:
        if child.type in ("identifier", "name", "property_identifier"):
            return child.text.decode("utf-8")
    return "<anonymous>"


def _extract_docstring(node, source_code: str, language: str) -> str | None:
    """Extract docstring from the first statement if it's a string literal."""
    body = None
    for child in node.children:
        if child.type in ("block", "statement_block", "class_body"):
            body = child
            break
    if body is None:
        return None

    for child in body.children:
        if child.type in ("expression_statement",):
            for gc in child.children:
                if gc.type in ("string", "concatenated_string"):
                    text = source_code[gc.start_byte:gc.end_byte]
                    # Strip triple quotes
                    for q in ('"""', "'''", '`', '"', "'"):
                        text = text.strip(q)
                    return text.strip()
        break  # only check first statement
    return None


def _extract_signature(content: str, language: str) -> str:
    """Extract the first line (signature) from a function/class definition."""
    first_line = content.split("\n")[0].rstrip()
    # Remove trailing colon for Python
    return first_line


def _extract_calls(node, source_code: str) -> list[str]:
    """Extract function call names from within a node."""
    calls = []
    _walk_calls(node, source_code, calls)
    return list(set(calls))[:20]  # cap to avoid noise


def _walk_calls(node, source_code: str, calls: list):
    """Recursively find call expressions."""
    if node.type == "call":
        fn = node.children[0] if node.children else None
        if fn:
            calls.append(source_code[fn.start_byte:fn.end_byte])
    for child in node.children:
        _walk_calls(child, source_code, calls)


def _extract_module_docstring(root, source_code: str, language: str) -> str | None:
    """Extract module-level docstring (first string expression in file)."""
    for child in root.children:
        if child.type == "expression_statement":
            for gc in child.children:
                if gc.type in ("string", "concatenated_string"):
                    text = source_code[gc.start_byte:gc.end_byte]
                    for q in ('"""', "'''"):
                        text = text.strip(q)
                    return text.strip()
            break
        elif child.type not in ("comment",):
            break
    return None


def _fallback_parse(file_path: Path, source_code: str, language: str) -> dict:
    """Regex-free fallback: treat the entire file as a single unit.
    Used for non-parseable files (markdown, config, etc.)."""
    return {
        "functions": [],
        "classes": [],
        "imports": [],
        "module_docstring": None,
    }
