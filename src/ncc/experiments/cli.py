"""Command line: `python -m ncc run ...` and `python -m ncc evaluate ...`, from any directory."""
import argparse
import logging

from ncc.experiments.config import load_experiment
from ncc.experiments.evaluate import evaluate
from ncc.experiments.run import run_experiment


def main(argv=None):
    """Parses argv (default: sys.argv[1:]) and runs the command."""
    args = _parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(levelname)s: %(message)s')
    exp = load_experiment(args.input)
    if args.command == 'run':
        seed = exp.seed if args.seed is None else args.seed
        run_experiment(exp, seed, device=args.device)
    else:
        evaluate(exp, seed=args.seed, all_seeds=args.multi_seeds, clip=args.clip, sort=args.sort)


def _parser():
    parser = argparse.ArgumentParser(prog='python -m ncc', description='Neural Conformal Control: paper experiments')
    commands = parser.add_subparsers(dest='command', required=True)

    run = commands.add_parser('run', help='train NCC and calibrate the test steps')
    run.add_argument('--input', '-i', required=True, help='experiment id, i.e. configs/experiments/<id>.yaml')
    run.add_argument('--seed', '-s', type=int, help='seed; defaults to the seed in the experiment config')
    run.add_argument('--device', help='torch device (e.g. cpu, cuda:0, mps); overrides ncc.device for this run')

    ev = commands.add_parser('evaluate', help='compute the metrics of saved results and write them to a CSV')
    ev.add_argument('--input', '-i', required=True, help='experiment id, i.e. configs/experiments/<id>.yaml')
    seeds = ev.add_mutually_exclusive_group()
    seeds.add_argument('--seed', '-s', type=int, help='seed to evaluate; defaults to the seed in the experiment config')
    seeds.add_argument('--multi_seeds', '-u', action='store_true', help='evaluate every seed in `seeds`')
    ev.add_argument('--clip', '-c', action='store_true', help='only evaluate the steps in data/<dataset>_clip_idx.pkl')
    ev.add_argument('--sort', '-o', action='store_true', help='sort the interval widths so they shrink as alpha grows')
    return parser
