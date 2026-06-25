# Paper Workflow

Codex maintains this directory as a repository-synchronized manuscript shell.
The manuscript should not be treated as the source of truth for experimental
numbers. Regenerate paper tables and the experiment log after each run:

```bash
python repro/scripts/sync_paper_artifacts.py
```

Compile from this directory:

```bash
make -C paper
```

If LaTeX packages are missing on the local machine, compile on the server or
install a TeX distribution. The generated tables use data from
`repro/diagnostics/`.
