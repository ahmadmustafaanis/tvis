"""The UI ships as plain ES modules (no bundler), so nothing else checks that imports resolve."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from tvis.viewer.server import ui_dir

UI = ui_dir()
JS = sorted((UI / "js").glob("*.js"))
IMPORT = re.compile(r'import\s*\{([^}]*)\}\s*from\s*"(\./[^"]+)"', re.S)
EXPORT = re.compile(r"export\s+(?:async\s+)?(?:function|const|let|class)\s+([A-Za-z_$][\w$]*)")


def exports_of(path: Path) -> set[str]:
    return set(EXPORT.findall(path.read_text()))


def test_ui_is_packaged_with_an_entry_point():
    html = (UI / "index.html").read_text()

    assert '<script type="module" src="/static/js/app.js">' in html
    assert (UI / "style.css").is_file()
    assert (UI / "js" / "app.js") in JS


@pytest.mark.parametrize("module", JS, ids=lambda p: p.name)
def test_every_named_import_resolves_to_an_export(module: Path):
    for names, target in IMPORT.findall(module.read_text()):
        target_path = (module.parent / target).resolve()
        assert target_path.is_file(), f"{module.name} imports missing {target}"
        wanted = {n.strip().split(" as ")[0] for n in names.split(",") if n.strip()}
        missing = wanted - exports_of(target_path)
        assert not missing, f"{module.name} imports {sorted(missing)} that {target} does not export"


def test_every_module_is_reachable_from_the_entry_point():
    reachable, todo = set(), [UI / "js" / "app.js"]
    while todo:
        path = todo.pop()
        if path in reachable:
            continue
        reachable.add(path)
        todo += [(path.parent / t).resolve() for _, t in IMPORT.findall(path.read_text())]

    assert {p.resolve() for p in JS} == reachable


def test_css_uses_only_declared_colour_tokens():
    css = (UI / "style.css").read_text()
    inline = {m for p in JS for m in re.findall(r'"(--[\w-]+)"\s*:', p.read_text())}  # style: {"--kc": ...}
    declared = set(re.findall(r"(--[\w-]+)\s*:", css)) | inline
    used = set(re.findall(r"var\((--[\w-]+)", css))
    js_used = {m for p in JS for m in re.findall(r"var\((--[\w-]+)", p.read_text())}
    dynamic = {u for u in js_used if u.endswith("-")}  # e.g. `var(--k-${kind})`

    assert used - declared == set()
    assert (js_used - dynamic) - declared == set()
