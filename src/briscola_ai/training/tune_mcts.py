from __future__ import annotations
import itertools, random, argparse
from briscola_ai.core.env import BriscolaEnv
from briscola_ai.agents.mcts import ISMCTSAgent
from briscola_ai.agents.rule_based import BestChoiceFirstAgent

def run_match(env, a0, a1, games=200, seed=0):
    rng = random.Random(seed)
    wins=[0,0]; pts=[0,0]
    for _ in range(games):
        # mescola i lati per evitare bias di mano
        if rng.random()<0.5:
            aa0, aa1 = a0, a1
            idx0 = 0
        else:
            aa0, aa1 = a1, a0
            idx0 = 1
        w, p = env.play_game(aa0, aa1)
        if w==0: wins[idx0]+=1
        elif w==1: wins[1-idx0]+=1
        pts[0]+=p[0]; pts[1]+=p[1]
    total=games
    return wins, (pts[0]/total, pts[1]/total)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    env = BriscolaEnv(seed=args.seed)
    grid_iters = [400, 800, 1200]
    grid_c = [0.2, 0.4, 0.7]
    grid_disc = [0.6, 0.8]

    print("TUNING vs BestChoiceFirst")
    best=None
    for iters, c, disc in itertools.product(grid_iters, grid_c, grid_disc):
        a_mcts = ISMCTSAgent(iterations=iters, c=c, discount=disc, seed=args.seed)
        a_rule = BestChoiceFirstAgent()
        wins, avg = run_match(env, a_mcts, a_rule, games=args.games, seed=args.seed)
        wr = wins[0]/args.games
        print(f"iters={iters:4d} c={c:.2f} disc={disc:.2f}  WR={wr:.3f}  avgPts={avg}")
        if not best or wr>best[0]:
            best=(wr,(iters,c,disc))
    print("Best:", best)

if __name__=="__main__":
    main()
