"""Command line entry point."""
from __future__ import annotations
import argparse
from .simulator import Simulation
from .evaluation import generate_plots

def main() -> None:
    parser=argparse.ArgumentParser(description="Run TokenOS/TokenFL simulator")
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--devices", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="results/")
    args=parser.parse_args()
    result=Simulation(args.rounds, args.devices, args.seed).run(args.output)
    generate_plots(result["metrics"], args.output)

if __name__ == "__main__":
    main()
