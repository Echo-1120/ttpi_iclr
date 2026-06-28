# PAM Formal Server Commands

Run these on the 5080 server from the repository root after pulling `repro/hardmove-5080`.

```bash
cd ~/code/ttpi_iclr
git pull origin repro/hardmove-5080
```

First validate that the stage-2 smoke run is complete:

```bash
python3 repro/scripts/check_pam_smoke_results.py \
  --manifest repro/experiments/pam_baseline_smoke_manifest.json
```

Generate the formal command manifest without starting training:

```bash
python3 repro/scripts/run_pam_ablation.py \
  --manifest repro/experiments/pam_formal_server_manifest.json
```

Start or resume the formal server run only after the smoke check passes:

```bash
python3 repro/scripts/run_pam_ablation.py \
  --manifest repro/experiments/pam_formal_server_manifest.json \
  --execute \
  --resume
```

After all training JSON files exist, generate spectra and figures:

```bash
python3 repro/scripts/run_pam_ablation.py \
  --manifest repro/experiments/pam_formal_server_manifest.json \
  --execute \
  --resume \
  --run-spectra \
  --make-plots
```

Do not use `--execute` on the formal manifest until the smoke validator prints `[OK]`.
