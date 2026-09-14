from pathlib import Path
import importlib.util

PROJECT_ROOT = Path(__file__).resolve().parents[1]

packages = [
    "duckdb",
    "pandas",
    "numpy",
    "sklearn",
    "pyarrow",
]

print(f"Project root: {PROJECT_ROOT}")
print("\nCore package check:")

failed = False
for package in packages:
    exists = importlib.util.find_spec(package) is not None
    print(f"  {'OK' if exists else 'MISSING':7} {package}")
    failed = failed or not exists

print("\nRequired data files:")
for name in ["train_transaction.csv", "train_identity.csv"]:
    path = PROJECT_ROOT / "data" / "raw" / name
    print(f"  {'FOUND' if path.exists() else 'MISSING':7} {path}")

if failed:
    raise SystemExit("\nInstall requirements before continuing.")

print("\nEnvironment looks ready.")
