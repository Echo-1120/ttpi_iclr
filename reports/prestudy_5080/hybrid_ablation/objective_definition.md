# Hybrid Ablation Objective Definition

- `hybrid_block_only`: block legality only; zero objective; local block order tie-break.
- `hybrid_block_plus_physics`: physical HardMove coupling objective.
- `hybrid_block_plus_sensitivity`: deterministic angular-distance sensitivity coupling objective.
- `hybrid_current`: `0.5 * physical + 0.5 * deterministic sensitivity`.

All variants preserve the TTPI training configuration and change only the action-mode ordering.

Loaded hybrid ablation result rows: 0.
