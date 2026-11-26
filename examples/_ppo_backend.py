"""Backend selection shared by the built-in PPO examples."""
import argparse
from functools import partial

from geneticevo import RustNativePPO


def add_backend_arguments(parser, *, allow_accelerated=False):
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--rust", action=argparse.BooleanOptionalAction, default=None,
                        help="Use Rust PPO (CPU default). Pass --no-rust for Torch PPO.")
    if allow_accelerated:
        parser.add_argument("--accelerated", action="store_true", help="Use accelerated Torch PPO")
        parser.add_argument("--chunk-size", type=int)
    else:
        parser.set_defaults(accelerated=False, chunk_size=None)


def ppo_options(parser, args, **options):
    use_rust = args.rust
    if use_rust is None:
        use_rust = args.device == "cpu" and not args.accelerated

    options.update(device=args.device, population_chunk_size=args.chunk_size)
    if use_rust:
        if args.device != "cpu" or args.accelerated:
            parser.error("--rust requires CPU and is incompatible with --accelerated")
        try:
            learner = RustNativePPO(**options)
        except ImportError as error:
            parser.error(f"{error} Pass --no-rust to use Torch PPO.")
        return dict(learner=learner, backend="rust")

    import torch
    from geneticevo.ppo import NativePPO
    from geneticevo.tensor import TensorReach1D

    torch.set_num_threads(1)
    return dict(learner=NativePPO(accelerated=args.accelerated, **options), backend="batched",
                tensor_env_factory=partial(TensorReach1D, continuous=args.continuous))
