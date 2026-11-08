"""Build the public code ZIP from an EXPLICIT allowlist and verify it. Usage: python tools/build_public_zip.py OUT.zip
Optional: --pack FOLDER also checks that no line of the confidential scripts appears in the ZIP."""
import re
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ALLOW = ["README.md", ".gitignore",
         "promofinder/__init__.py", "promofinder/__main__.py", "promofinder/parser.py", "promofinder/validate.py",
         "promofinder/scoring.py", "promofinder/llm.py", "promofinder/pipeline.py", "promofinder/synthetic.py",
         "promofinder/demo.py",
         "tests/_common.py", "tests/test_validator.py", "tests/test_parser.py", "tests/test_llm.py", "tests/test_pipeline.py",
         "tools/build_public_zip.py", "tools/bench_scaling.py"]
SECRET_PATTERNS = [r"AIza[0-9A-Za-z_\-]{20,}", r"sk-[A-Za-z0-9]{20,}", r"sk-ant-[A-Za-z0-9_\-]{20,}", r"-----BEGIN [A-Z ]*PRIVATE KEY",
                   r"(?i)api[_-]?key\s*=\s*['\"][A-Za-z0-9_\-]{16,}"]


def main():
    args = sys.argv[1:]
    pack = None
    if "--pack" in args:
        i = args.index("--pack")
        pack = Path(args[i + 1])
        del args[i:i + 2]
    out = Path(args[0]) if args else ROOT.parent / "public_code.zip"
    missing = [f for f in ALLOW if not (ROOT / f).is_file()]
    if missing:
        sys.exit(f"allowlisted files missing: {missing}")
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in ALLOW:
            z.write(ROOT / f, f"promofinder_repo/{f}")
    names = sorted(zipfile.ZipFile(out).namelist())
    assert names == sorted(f"promofinder_repo/{f}" for f in ALLOW), "ZIP contents differ from the allowlist"
    text = {n: zipfile.ZipFile(out).read(n).decode("utf-8", "replace") for n in names}
    for n, t in text.items():
        for pat in SECRET_PATTERNS:
            if re.search(pat, t):
                sys.exit(f"possible credential in {n}: {pat}")
    if pack:
        blob = "\n".join(text.values()).lower()
        hits = 0
        for f in sorted(pack.rglob("*.txt")):
            for ln in f.read_text(encoding="utf-8", errors="replace").splitlines():
                ln = ln.strip().strip('"')
                if len(ln) >= 30 and not ln.startswith(("**", "Start:", "Scene", "EPISODE", "CONTINUITY")) and ln.lower() in blob:
                    hits += 1
                    print("CONFIDENTIAL LINE FOUND:", ln[:60])
        if hits:
            sys.exit(f"{hits} confidential script line(s) found in the ZIP")
        print("confidential-script check: no script line (>=30 chars) of the pack appears in the ZIP")
    print(f"wrote {out} with {len(names)} files:")
    for n in names:
        print("  ", n)


if __name__ == "__main__":
    main()
