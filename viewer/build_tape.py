#!/usr/bin/env python3
"""Pack recorded games into the self-contained Game Tape page.

    PYTHONPATH=. python3 viewer/build_tape.py SPEC.json --out viewer/game-tape.html

SPEC is a JSON object: {"footer": "...", "reels": [{"id", "label", "blurb",
"note": [paragraphs], "a", "b", "names": [A, B], "steps", "seed", "cfg"}, ...]},
where "a" and "b" are anything hockey.watch.resolve_policy accepts (a
checkpoint path, "chase", "sniper", "random", "still") and the optional "cfg"
holds Config overrides for that reel, e.g. {"goalie": true, "solid_net": true}.

Each game is recorded with hockey.export.record -- the same trajectory the
Godot viewer replays -- then packed as nine floats per frame (both skaters'
x, y, heading; the puck's x, y; the possessor), plus both goalies' x, y in a
goalie game (13 per frame, the reel's "stride"), so a minute is ~60-90 KB.
The page needs no server: open the output file in a browser, or publish it.
"""
import argparse, json, os, sys
from dataclasses import replace

from hockey.config import DEFAULT

from hockey.export import record
from hockey.watch import resolve_policy

HERE = os.path.dirname(os.path.abspath(__file__))


def pack(traj):
    f, goals = [], []
    for i, fr in enumerate(traj["frames"]):
        (ax, ay, at), (bx, by, bt) = fr["skaters"]
        px, py = fr["puck"]
        f += [round(ax, 2), round(ay, 2), round(at, 3), round(bx, 2), round(by, 2),
              round(bt, 3), round(px, 2), round(py, 2), int(fr["possessor"])]
        if "goalies" in fr:
            f += [round(c, 2) for xy in fr["goalies"] for c in xy]
        ev = fr.get("event")
        if ev in ("goal_a", "goal_b"):
            goals.append([i, 0 if ev == "goal_a" else 1])
    return f, goals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("spec")
    ap.add_argument("--template", default=os.path.join(HERE, "tape", "template.html"))
    ap.add_argument("--out", default=os.path.join(HERE, "game-tape.html"))
    args = ap.parse_args()
    spec = json.load(open(args.spec))

    tape = {"fps": None, "rink": None, "footer": spec.get("footer", ""), "reels": []}
    for r in spec["reels"]:
        pa, _ = resolve_policy(r["a"])
        pb, _ = resolve_policy(r["b"])
        cfg = replace(DEFAULT, **r.get("cfg", {}))
        traj = record(pa, pb, steps=r.get("steps", 1800), seed=r.get("seed", 0),
                      cfg=cfg, names=tuple(r["names"]))
        f, goals = pack(traj)
        tape["fps"] = traj["fps"]
        tape["rink"] = {k: v for k, v in traj["rink"].items()
                        if k not in ("solid_net", "goalie_radius")}
        tape["reels"].append({
            "stride": 13 if cfg.goalie else 9,
            "goalie_radius": cfg.goalie_radius if cfg.goalie else None,
            "id": r["id"], "label": r["label"], "blurb": r["blurb"], "note": r["note"],
            "teams": [{"name": t["name"], "colour": t["colour"]} for t in traj["teams"]],
            "final": traj["final_score"], "goals": goals, "f": f,
        })
        print(f"  {r['id']:<12} {r['names'][0]} {traj['final_score'][0]}-"
              f"{traj['final_score'][1]} {r['names'][1]}  ({len(traj['frames'])} frames)", flush=True)

    blob = json.dumps(tape, separators=(",", ":")).replace("</", "<\\/")
    html = open(args.template).read()
    if "/*TAPE_JSON*/null" not in html:
        sys.exit("template is missing the /*TAPE_JSON*/null placeholder")
    with open(args.out, "w") as fh:
        fh.write(html.replace("/*TAPE_JSON*/null", blob, 1))
    print(f"wrote {args.out} ({os.path.getsize(args.out) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
