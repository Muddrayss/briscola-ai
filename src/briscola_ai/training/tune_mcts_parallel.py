from __future__ import annotations
import argparse, itertools, os
from concurrent.futures import ProcessPoolExecutor, as_completed

from briscola_ai.core.env import BriscolaEnv
from briscola_ai.agents.mcts import ISMCTSAgent
from briscola_ai.agents.rule_based import BestChoiceFirstAgent

def eval_cfg(cfg, games: int, seed: int):
    iters, c, disc, t = cfg
    env = BriscolaEnv(seed=seed)
    mcts = ISMCTSAgent(iterations=iters, c=c, discount=disc, seed=seed, prior_temp=t)
    rule = BestChoiceFirstAgent()
    wins=[0,0]; pts=[0,0]
    import random
    rng = random.Random(seed)
    for _ in range(games):
        if rng.random()<0.5:
            a0,a1 = mcts,rule; idx0=0
        else:
            a0,a1 = rule,mcts; idx0=1
        w,p = env.play_game(a0,a1)
        if w==0: wins[idx0]+=1
        elif w==1: wins[1-idx0]+=1
        pts[0]+=p[0]; pts[1]+=p[1]
    WR = wins[0]/games
    return (WR, (iters,c,disc,t), (pts[0]/games, pts[1]/games))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=60)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    args = ap.parse_args()

    grid_iters = [400, 800, 1200]
    grid_c     = [0.5, 0.8, 1.2]      # PUCT usa c più alto di UCT
    grid_disc  = [0.6, 0.8]
    grid_temp  = [0.8, 1.0, 1.5]      # “morbidezza” dei prior

    cfgs = list(itertools.product(grid_iters, grid_c, grid_disc, grid_temp))
    print(f"TUNING PARALLELO vs BestChoiceFirst  |  configs={len(cfgs)}  workers={args.workers}")

    best=None
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(eval_cfg, cfg, args.games, args.seed) for cfg in cfgs]
        for fu in as_completed(futs):
            WR, params, avg = fu.result()
            iters,c,disc,t = params
            print(f"iters={iters:4d} c={c:.2f} disc={disc:.2f} temp={t:.2f}  WR={WR:.3f}  avgPts={avg}")
            if not best or WR>best[0]:
                best=(WR, params, avg)
    print("\nBest:", best)

if __name__=="__main__":
    main()
