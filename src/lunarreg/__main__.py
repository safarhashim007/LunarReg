import argparse


def main():
    parser = argparse.ArgumentParser(
        description=(
            "LunarReg: image-only V9 global registration, "
            "optional experimental V11. "
            "Output directory must not exist."
        )
    )

    sub = parser.add_subparsers(
        dest="command",
        required=True
    )

    p = sub.add_parser("register")

    p.add_argument("--source", required=True)
    p.add_argument("--reference", required=True)
    p.add_argument("--output", required=True)

    p.add_argument("--source-geotiff")
    p.add_argument("--reference-geotiff")

    p.add_argument(
        "--mode",
        choices=["fast", "base", "precise"],
        default="fast"
    )

    p.add_argument(
        "--device",
        choices=["auto", "cpu", "mps", "cuda"],
        default="auto"
    )

    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--registration-mode", choices=["legacy", "blind"], default="legacy",
                   help="legacy: V9/TMC priors (default); blind: full rotation, image-only ranking and gate, source-pixel cycles")
    p.add_argument("--blind-rotation-step", type=float, default=30.0,
                   help="Blind mode only: source rotation spacing in degrees across [0, 360) (default: 30)")

    p.add_argument(
        "--resume-blind",
        action="store_true",
        help="Resume a blind rotation sweep from saved rotation match files"
    )
    p.add_argument("--expected-scale", type=float, help="Legacy mode only")
    p.add_argument(
        "--expected-rotation",
        type=float,
        default=None
    )

    p.add_argument(
        "--no-local-refine",
        action="store_true"
    )

    p.add_argument(
        "--experimental-local",
        action="store_true"
    )

    p.add_argument(
        "--frozen-baseline",
        help=(
            "Explicit replay of saved pair V9; "
            "input hashes must match original pair"
        )
    )

    b = sub.add_parser("benchmark")

    b.add_argument("--source", required=True)
    b.add_argument("--output", required=True)
    b.add_argument("--seed", type=int, default=42)

    b.add_argument(
        "--device",
        choices=["auto", "cpu", "mps", "cuda"],
        default="auto"
    )

    args = parser.parse_args()

    if args.command == "benchmark":
        from .benchmark import run_benchmark
        return run_benchmark(args)

    if args.command == "register":
        if args.registration_mode == "blind" and (args.expected_scale is not None or args.expected_rotation is not None or args.frozen_baseline):
            parser.error("Blind mode cannot use expected scale/rotation or a frozen legacy baseline")
        if args.registration_mode == "blind" and not 0 < args.blind_rotation_step <= 180:
            parser.error("Blind rotation step must be greater than 0 and at most 180 degrees")
        # Heavy imports such as torch/RoMa happen only
        # when registration is actually requested.
        from .pipeline import run
        return run(args)

    parser.error("Unknown command")


if __name__ == "__main__":
    raise SystemExit(main())
