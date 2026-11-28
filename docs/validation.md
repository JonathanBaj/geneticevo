# Correctness and performance validation

Run the suite and non-training benchmarks from the repository:

```sh
python -m pytest -q
python benchmarks/equal_work.py --repeats 3
python benchmarks/baselines.py --seeds 1 2 3
```

The tests cover selection, elites, mutation, population counts, lineage, archive
independence, frozen evaluation, and exact checkpoint resume. They also verify
width, depth, and graph mutations, portable agent saves, and state isolation
for custom learners in Lamarckian and Baldwinian runs. Process evaluation uses
spawned workers and needs permission to create worker pipes.

The optional Torch and Rust tests compare inference against NumPy with discrete
and bounded continuous actions. Rust tests include mixed-width and mixed-depth
MLPs and sparse graphs within the same population. Frozen episode returns,
lengths, and interaction counts are checked against the Gymnasium reference.
The CUDA test runs with a usable GPU and checks variable-width evaluation with
both native tensor and Gymnasium environments.

Shared-world tests exercise routing, message timing, communication costs, masks,
rotating partners, and archives evaluated against a fixed partner pool.
The PettingZoo integration also runs the upstream Parallel API test.

`equal_work.py` compares serial, process, and Torch-batched evaluation over
equal interaction counts. It records wall time for full generations, including
initialization, reproduction, frozen evaluation, and archive evaluation. CUDA
variants run on machines with a usable GPU. Timings exclude imports and warm-up.

`baselines.py` compares random and evolved policies on a fixed final test suite
across seeds. Its separate communication ablation uses a different task. The
results are descriptive and include the actual interaction counts.
