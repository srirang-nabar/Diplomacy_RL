# Shipped checkpoints (CLAUDE.md §7.1)

Every file here is a plain state_dict payload: load with
`triad.rl.checkpoint.load_policy(path, device="cpu")` — CPU-only,
`weights_only=True`, works on any machine. Verified in CI by
`tests/test_checkpoint.py` (SHA256 + load + full game).

| file | sha256 | producing config | seed | training git SHA | date | notes |
|------|--------|------------------|------|------------------|------|-------|
| bc.pt | `dabad84aa22f0fc7d31c38990ef314dde38e5843955ce1d8762796c71507486b` | MLP 2x256 + GRU-256 AR decoder; BC on 50k mixed-bot games (Grabber-heavy), 1.25M samples, 3 epochs, lr 1e-3, batch 512; own-frame decode order (post seat-equivariance fix) | 0 | 772786d | 2026-07-04 | val CE 0.728, top-1 0.612. Acceptance: 100% solo vs 2xRandomLegal (argmax, 500g); 33.2% sampled vs 2xGrabber (500g) — exact 1/3 teacher parity; the pre-fix 26.0% shortfall was the ordering bug, not a BC ceiling |
| ppo_kl05.pt | `df8cbb692d579887bea9eb81f7e8769dfac183ee2d619eda591efccfda20eb7e` | BC-init + PPO self-play, beta_kl=0.05, alpha=0.02, 500k env-steps (local standard budget), configs/triad.yaml; own-frame decode order (post seat-equivariance fix) | 0 | 772786d | 2026-07-04 | M3 final. 500g evals vs 2xGrabber: 71.6% sampled / 72.4% argmax (BC: 33.2/17.8); 100% vs 2xRandom. Seat chi2 re-verified: pooled 600g p=0.099, no stable ordering (pre-fix: p~1e-4, C>A>B). In-training eval peaked ~0.93 at 80k steps, settled ~0.73 |
