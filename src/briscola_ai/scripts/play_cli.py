from __future__ import annotations
import argparse, random
from briscola_ai.core.env import BriscolaEnv
from briscola_ai.agents.rule_based import RandomAgent, HighestFirstAgent, BestChoiceFirstAgent

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    env = BriscolaEnv(seed=args.seed)
    agents = [
        ("BestChoiceFirst", BestChoiceFirstAgent()),
        ("HighestFirst",   HighestFirstAgent()),
    ]

    wins = [0,0]
    pts = [0,0]
    for _ in range(args.games):
        random.shuffle(agents)
        (n0,a0),(n1,a1) = agents
        winner, points = env.play_game(a0, a1, verbose=False)
        if winner == 0: wins[0] += 1
        elif winner == 1: wins[1] += 1
        pts[0] += points[0]; pts[1] += points[1]

    total = args.games
    print(f"Matchup: {agents[0][0]} (P0) vs {agents[1][0]} (P1)")
    print(f"Wins: P0={wins[0]}  P1={wins[1]}  Draws={total-wins[0]-wins[1]}")
    print(f"Avg points: P0={pts[0]/total:.1f}  P1={pts[1]/total:.1f}")

if __name__ == '__main__':
    main()
