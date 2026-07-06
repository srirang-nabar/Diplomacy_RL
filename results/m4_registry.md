# M4 run registry

All runs: 500k env-steps, population play (0.8 latest / 0.2 pool), seed 0.
Evals: 500g sampled vs 2xGrabber, 300g argmax vs 2xRandom, Wilson CIs in the notebook.

| run | beta | alpha | init | vs grabber | vs random | best-snap grabber | chi2 p | wall (min) |
|-----|------|-------|------|------------|-----------|-------------------|--------|------------|
| kl05_a02 | 0.05 | 0.02 | anchor | 0.734 | 1.000 | 0.916@163840 | 0.3679 | 32.1 |
| kl0_a02 | 0.0 | 0.02 | anchor | 0.924 | 1.000 | 0.924@491520 | 0.3829 | 25.2 |
| kl01_a02 | 0.01 | 0.02 | anchor | 0.764 | 1.000 | 0.808@245760 | 0.0626 | 31.6 |
| kl20_a02 | 0.2 | 0.02 | anchor | 0.788 | 1.000 | 0.790@245760 | 0.1984 | 33.0 |
| scratch_a02 | 0.0 | 0.02 | scratch | 0.028 | 0.993 | 0.056@245760 | 0.2404 | 22.8 |
| kl05_a00 | 0.05 | 0.0 | anchor | 0.688 | 1.000 | 0.908@163840 | 0.5407 | 36.4 |
| kl0_a00 | 0.0 | 0.0 | anchor | 0.926 | 1.000 | 0.894@163840 | 0.6575 | 27.8 |
| scratch_a00 | 0.0 | 0.0 | scratch | 0.136 | 0.997 | 0.150@409600 | 1.0 | 26.0 |
