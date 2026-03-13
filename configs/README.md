# Configs Index

## Primary entry points

- `mvp_mbpp.yaml`: default MBPP experiment preset.
- `mvp_humaneval.yaml`: default HumanEval experiment preset.
- `mvp_mbpp_rigorous.yaml`: larger MBPP preset for full evaluations.

## Tuned / comparison presets

- `mvp_mbpp_eig_tuned.yaml`: tuned EIG parameters (MBPP).
- `mvp_mbpp_eig_tuned_smoke.yaml`: smaller tuned EIG smoke preset.
- `mvp_mbpp_eig_gap_tuned.yaml`: tuned for stronger EIG-vs-baseline gap.
- `mvp_mbpp_eig_vs_random_tuned.yaml`: focused EIG vs random comparison.
- `mvp_mbpp_eig_vs_random_tuned_full.yaml`: full-size tuned EIG vs random comparison.
- `mvp_humaneval_eig_vs_random_full.yaml`: full HumanEval EIG vs random comparison.

## Ablation presets

- `mvp_mbpp_ab_smoke.yaml`: small ablation smoke config.
- `mvp_mbpp_ab_quick.yaml`: quick ablation config.
- `mvp_mbpp_ticoder_ab.yaml`: TiCoder scoring/update ablation baseline.
- `mvp_mbpp_ticoder_ab_quick1.yaml`: quick TiCoder ablation with 1 example.
- `mvp_mbpp_ticoder_ab_quick5.yaml`: quick TiCoder ablation with 5 examples.
- `mvp_mbpp_ticoder_ab_quick10.yaml`: quick TiCoder ablation with 10 examples.

## Other

- `mvp_humaneval_google_oneshot.yaml`: HumanEval one-shot profile tuned for Gemini wrapper.
- `model_profiles.yaml`: named model profile matrix used by `run_model_matrix.py`.

## Conventions

- File names use `mvp_<dataset>_<purpose>.yaml`.
- All paths are repo-relative.
- Keep strategy knobs under `pipeline` and model endpoint vars under `model`.
