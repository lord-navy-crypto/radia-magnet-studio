# Uploading This Project to GitHub

## Before publishing

1. Review `README.md`, `LICENSE`, `RADIA_LICENSE.txt`, and
   `THIRD_PARTY_NOTICES.md`.
2. Confirm that no passwords, API keys, tokens, personal datasets, or private
   paths are present.
3. Do not add a locally compiled `radia*.so` unless you intentionally choose to
   distribute a binary and include all required RADIA license materials.
4. Run the tests:

   ```bash
   python3 tests/run_all.py
   ```

## First upload from macOS Terminal

Create an empty repository on GitHub without automatically adding a README,
license, or `.gitignore`. Then open Terminal in this folder and run:

```bash
git init -b main
git add .
git status
git commit -m "Initial public release"
git remote add origin https://github.com/YOUR-USERNAME/radia-magnet-studio.git
git push -u origin main
```

Replace `YOUR-USERNAME` with the actual GitHub username. Inspect the output of
`git status` before committing so that generated files or private information
are not accidentally published.

## Suggested repository description

> Independent Streamlit interface and analysis toolkit for RADIA-based
> insertion-device modelling, with macOS-oriented launch support, engineering
> error analysis, visualization, and scientific-data export.
