# Shipped checkpoints (CLAUDE.md §7.1)

Every file here is a plain state_dict payload: load with
`triad.rl.checkpoint.load_policy(path, device="cpu")` — CPU-only,
`weights_only=True`, works on any machine. Verified in CI by
`tests/test_checkpoint.py` (SHA256 + load + full game).

| file | sha256 | producing config | seed | training git SHA | date | notes |
|------|--------|------------------|------|------------------|------|-------|
| bc.pt | `2cfefdd4fae4dd2521b7641e1c9c6dcc738ac7d26180db0c3c9bddca60f0241b` | MLP 2x256 + GRU-256 AR decoder; BC on 50k mixed-bot games (Grabber-heavy), 1.25M samples, 3 epochs, lr 1e-3, batch 512 | 0 | 12c1d0d-dirty | 2026-07-04 | val CE 0.732, top-1 0.611. Acceptance: 99.6% solo vs 2xRandomLegal (argmax, 500g); vs 2xGrabber 23.4% argmax / 26.0% sampled (500g) — slightly below the 1/3 teacher parity, documented BC ceiling |
