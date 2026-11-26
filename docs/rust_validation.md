# Rust CPU PPO validation

## Baseline profile

The starting checkout was 0.1.0a7. The brief described the earlier width-only
release. This checkout also supported shrinking, adding and removing hidden
layers, and sparse graph policies with recurrent connections. The Rust path
uses those same architecture edits and stable identities.

The pre-port cProfile run used 32 agents, two 16-unit hidden layers, two
generations, two environment copies, eight steps per rollout, two PPO epochs,
and 64 training transitions per member per generation. Width and depth edits
were enabled. All Torch runs used four CPU threads. Times below are seconds
over both generations and include profiler overhead.

| Phase | Native PPO with Gym | Accelerated PPO with Gym | Accelerated PPO with tensor simulation |
| --- | ---: | ---: | ---: |
| Complete profiled run | 7.004 | 1.307 | 1.156 |
| Learning setup | 0.258 | 0.086 | 0.066 |
| Rollout collection | 1.761 | 0.135 | 0.080 |
| PPO update, including forward and backward | 2.435 | 0.159 | 0.172 |
| Architecture checks, exports and edits | 0.709 | 0.339 | 0.321 |
| Final optimizer and parameter export | 0.047 | 0.033 | 0.033 |
| Frozen evaluation | 1.346 | 0.206 | 0.105 |
| Reproduction and archive work | 0.247 | 0.233 | 0.252 |

The Gym `Reach1D.step` method took 0.186 s cumulatively with native PPO and
0.110 s with accelerated PPO, across training and evaluation. The native
Gym/Torch bridge took 0.545 s and 0.139 s respectively, including environment
steps. Policy forward calls accounted for 1.609 s cumulatively in the native
case and 0.109 s in the Gym accelerated case. These call-level measurements
overlap the phases in the table. The accelerated path spent more time checking
and exporting architectures than running PPO updates. The Rust path keeps
active shapes in native memory and exports individual genes when an edit occurs.

This command profiles the same workload shape and prints the PPO phase totals
and cProfile call breakdown. The benchmark also enables offspring depth edits.

```sh
python benchmarks/rust_vs_python.py --populations 32 --widths 16 --variants native accelerated tensor rust --profile
```

In a single-seed run of this benchmark profile, autograd backward accounted
for 0.483 s with native Gym PPO, 0.055 s with accelerated Gym PPO, and 0.051 s
with accelerated tensor PPO. The full Rust PPO update phase took 0.008 s over
both generations, and Rust rollout collection took 0.081 s with Gymnasium.

## Benchmark method

Measured on an AMD Ryzen 9 8945HS (8 cores, 16 logical processors) on Windows
with CPython 3.11.9, NumPy 2.4.4, Gymnasium 1.2.3, PyTorch 2.12.1+cu126
running on CPU, and a Rust 1.85.1 release wheel. Both engines used four worker
threads. Each configuration ran one warm-up followed by complete Evolution
runs for seeds 42, 43 and 44. The growing matrix starts with two equal-width
hidden layers, uses the baseline rollout budget, and enables widening,
shrinking, hidden-layer insertion/removal and architecture mutation.
Complete-generation time includes initialization, learning, frozen evaluation,
reproduction and archives. Rust and the accelerated Gym comparator use the
same Gymnasium environment. The tensor column uses Torch's vectorized
`TensorReach1D` for simulation as well as Torch for policy and PPO math.
The engines use different action random streams.

```sh
python benchmarks/rust_vs_python.py --populations 16 128 --widths 4 16 64 256 --variants rust accelerated tensor
python benchmarks/rust_vs_python.py --populations 1024 --widths 16 64 --variants rust accelerated
python benchmarks/rust_vs_python.py --populations 4 --widths 4 --generations 3 --variants rust native accelerated tensor --seeds 42 43 44 45 46
```

## Growing networks

Each time cell shows median seconds per generation, with learning time in
parentheses. Width is the starting size of each of the two hidden layers.

| Agents | Width | Rust with Gym | Accelerated Torch with Gym | Accelerated Torch with tensor simulation | Rust speedup over Torch with Gym |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 4 | 0.064 (0.027) | 0.273 (0.185) | 0.237 (0.177) | 4.25× |
| 16 | 16 | 0.066 (0.028) | 0.278 (0.203) | 0.243 (0.177) | 4.25× |
| 16 | 64 | 0.070 (0.033) | 0.311 (0.216) | 0.282 (0.213) | 4.43× |
| 16 | 256 | 0.190 (0.104) | 0.712 (0.573) | 0.715 (0.579) | 3.74× |
| 128 | 4 | 0.382 (0.210) | 1.337 (0.949) | 1.085 (0.837) | 3.50× |
| 128 | 16 | 0.359 (0.196) | 1.282 (0.889) | 1.115 (0.819) | 3.57× |
| 128 | 64 | 0.465 (0.261) | 1.611 (1.143) | 1.373 (1.041) | 3.47× |
| 128 | 256 | 1.250 (0.695) | 5.184 (4.241) | 4.982 (4.129) | 4.15× |
| 1,024 | 16 | 2.827 (1.479) | 9.549 (6.332) | — | 3.38× |
| 1,024 | 64 | 4.215 (2.219) | 13.774 (9.739) | — | 3.27× |

The 1,024-agent runs performed 131,072 training transitions and 16,384
individual PPO updates per run for each engine. In the 1,024-agent, width-64
case, Rust reduced median learning time from 9.739 to 2.219 seconds per
generation. Timing includes Gym stepping, growth and export.

## Fixed networks

With growth disabled, a 128-agent comparison used the same budget, seeds and
worker count. Times are median seconds per generation with learning in parentheses.

| Width | Rust with Gym | Accelerated Torch with Gym | Accelerated Torch with tensor simulation |
| ---: | ---: | ---: | ---: |
| 16 | 0.273 (0.122) | 0.914 (0.398) | 0.629 (0.286) |
| 64 | 0.265 (0.124) | 0.947 (0.459) | 0.825 (0.374) |

```sh
python benchmarks/rust_vs_python.py --fixed --populations 128 --widths 16 64 --variants rust accelerated tensor
```

The measured configurations used a lightweight reference environment. For
environments with expensive Python steps, those steps still contribute to the
Rust backend's wall time.

## Fitness across seeds

A growth-example-sized workload used four agents, two four-unit hidden layers,
three generations and 64 training transitions per agent per generation. Higher
fitness is better. The table reports the fixed archive reference fitness of
each champion and the median final-population mean.

| Engine | Seed 42 | Seed 43 | Seed 44 | Seed 45 | Seed 46 | Median champion | Median final population mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Rust with Gym | -1.890 | -1.643 | -0.933 | -1.125 | -3.720 | -1.643 | -42.478 |
| Native Torch with Gym | -1.667 | -1.603 | -0.926 | -1.089 | -3.005 | -1.603 | -16.356 |
| Accelerated Torch with Gym | -1.667 | -19.666 | -0.933 | -1.125 | -3.871 | -1.667 | -50.444 |
| Accelerated Torch with tensor simulation | -1.667 | -19.666 | -0.933 | -1.125 | -3.871 | -1.667 | -56.795 |

Champion medians were close in this short run. Final-population means varied
substantially across engines. The PPO reference test separately verifies an
optimizer update on identical samples, and an observation-dependent bandit test
checks that training learns a policy different from the initial one.

## Verification

The Windows run passed 122 Python tests with 29 skips. The five Rust unit
tests, formatting checks, and Clippy with warnings denied also passed. A
separate Torch-free environment ran continuous PPO with growing 128-unit
MLPs and passed the existing core tests. The stable-ABI Windows wheel and
the pure-Python distributions passed the package metadata check.

```sh
cargo test --manifest-path rust/Cargo.toml --locked
cargo clippy --manifest-path rust/Cargo.toml --locked --all-targets -- -D warnings
python -m pytest -q tests/test_rust_ppo.py
```
