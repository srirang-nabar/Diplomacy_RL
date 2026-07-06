# Shipped checkpoints (CLAUDE.md §7.1)

Every file here is a plain state_dict payload: load with
`triad.rl.checkpoint.load_policy(path, device="cpu")` — CPU-only,
`weights_only=True`, works on any machine. Verified in CI by
`tests/test_checkpoint.py` (SHA256 + load + full game).

| file | sha256 | producing config | seed | training git SHA | date | notes |
|------|--------|------------------|------|------------------|------|-------|
| bc.pt | `d6a98a2adaf22e410f6bee8138075d6be74bff6556028327af5ce8173b540bc7` | MLP 2x256 + GRU-256 AR decoder; BC on 50k mixed-bot games, 1.25M samples, 3 epochs; FULLY own-frame interface (obs + action ids, post M4 action-frame fix) | 0 | uncommitted-m4-fix | 2026-07-05 | eval numbers in tasks.md M2 record |
| ppo_kl05.pt | `9c83456ddf7a29210a36cc94af6da17da5e5009a1128e96e0b91704470f6947f` | BC-init + PPO, beta_kl=0.05, alpha=0.02, 500k steps, population play; fully own-frame interface (post M4 action-frame fix) | 0 | uncommitted-m4-fix | 2026-07-05 | recipe row (BC+PPO+KL). 500g: 73.4% sampled vs 2xGrabber, chi2 p=0.37. In-training peak 0.916@163k (pool snapshot kept in runs/) |
| ppo_final.pt | `a40b79730ae2d806044216d9f49cbc0b91efd498f664787fe053cfa488a9fb75` | BC-init + PPO, beta_kl=0, alpha=0.02, 500k steps, population play; own-frame interface | 0 | uncommitted-m4-fix | 2026-07-05 | SELECTED FINAL AGENT (M4 grid winner): 92.4% sampled vs 2xGrabber (500g), 100% vs 2xRandom, chi2 p=0.38, self-play decisive (0 draws/150) |
