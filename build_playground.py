"""Build playground.html by inlining protect.py into a template.

Run: python build_playground.py
Output: playground.html in the same directory.
"""
import pathlib

HERE = pathlib.Path(__file__).parent
PROTECT_SRC = (HERE / "protect.py").read_text()
TEMPLATE = (HERE / "playground.template.html").read_text()

# Sanity check — protect.py must not contain </script> or it will break the embed.
assert "</script>" not in PROTECT_SRC, "protect.py contains </script>; cannot inline"

# Substitute the marker. We use a literal placeholder to avoid f-string conflicts
# with braces in the template.
PLACEHOLDER = "{{PROTECT_PY_SOURCE}}"
assert PLACEHOLDER in TEMPLATE, f"placeholder {PLACEHOLDER!r} not in template"

out = TEMPLATE.replace(PLACEHOLDER, PROTECT_SRC)
(HERE / "playground.html").write_text(out)
print(f"wrote playground.html ({len(out):,} chars)")
