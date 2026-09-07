import argparse
from .pipeline import run


def main():
    parser=argparse.ArgumentParser(description='LunarReg: image-only V9 global registration, optional experimental V11. Output directory must not exist.')
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('register')
    p.add_argument('--source',required=True);p.add_argument('--reference',required=True);p.add_argument('--output',required=True)
    p.add_argument('--source-geotiff');p.add_argument('--reference-geotiff')
    p.add_argument('--mode',choices=['fast','base','precise'],default='fast')
    p.add_argument('--device',choices=['auto','cpu','mps','cuda'],default='auto')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--expected-scale',type=float);p.add_argument('--expected-rotation',type=float,default=0)
    p.add_argument('--no-local-refine',action='store_true');p.add_argument('--experimental-local',action='store_true')
    p.add_argument('--frozen-baseline',help='Explicit replay of saved pair V9; input hashes must match original pair')
    b=sub.add_parser('benchmark')
    b.add_argument('--source',required=True);b.add_argument('--output',required=True);b.add_argument('--seed',type=int,default=42)
    b.add_argument('--device',choices=['auto','cpu','mps','cuda'],default='auto')
    args=parser.parse_args()
    if args.command=='benchmark':
        from .benchmark import run_benchmark
        return run_benchmark(args)
    return run(args)

if __name__=='__main__':
    raise SystemExit(main())
