import importlib.metadata
import sys

requirements = [
    ("langchain", "0.2.0"),
    ("langgraph", "0.6.7"),
    ("langchain-openai", "0.1.0"),
    ("langchain-core", "0.2.0"),
    ("pydantic", "2.0.0"),
    ("python-dotenv", "1.0.0"),
    ("openai", "1.0.0"),
    ("print-color", "0.4.6"),
    ("langchain-google-genai", "1.0.0"),
    ("chromadb", "0.4.20"),
    ("langchain-chroma", "0.1.0"),
    ("langgraph-checkpoint-sqlite", "1.0.0"),
    ("pypdf", "4.0.0"),
    ("python-docx", "1.1.0"),
    ("streamlit", "1.30.0"),
]

def parse_version(v):
    return tuple(int(x) for x in v.split(".")[:3])

ok, missing, outdated = [], [], []

for package, min_version in requirements:
    try:
        installed = importlib.metadata.version(package)
        if parse_version(installed) >= parse_version(min_version):
            ok.append((package, installed))
        else:
            outdated.append((package, installed, min_version))
    except importlib.metadata.PackageNotFoundError:
        missing.append((package, min_version))

print(f"\n{'='*50}")
print(f"Python: {sys.version.split()[0]}")
print(f"{'='*50}\n")

if ok:
    print(f"OK ({len(ok)})")
    for name, ver in ok:
        print(f"   {name:<35} {ver}")

if outdated:
    print(f"\nOUTDATED ({len(outdated)}) - needs upgrade")
    for name, installed, required in outdated:
        print(f"   {name:<35} installed={installed}  required>={required}")

if missing:
    print(f"\nMISSING ({len(missing)}) - not installed")
    for name, required in missing:
        print(f"   {name:<35} required>={required}")

if missing or outdated:
    print(f"\nFix with:\n   pip install -r requirements.txt")
else:
    print(f"\nAll packages installed and up to date.")
