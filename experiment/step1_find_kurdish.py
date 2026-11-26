"""Step 1: record activations, find Kurdish-reacting layers and neurons, build the Kurdish direction.

Run:  python experiment/step1_find_kurdish.py
Writes: results/layer_scores.json, results/top_neurons.json, results/directions.npz,
        results/scatter_points.json, results/step1_summary.txt
"""
import json
import os

import numpy as np

import sentences as S
from common import N_LAYERS, RESULTS, load_model, record

rng = np.random.default_rng(0)


def cohens_d(a, b):
    pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2) + 1e-8
    return float((a.mean() - b.mean()) / pooled)


def auc(a, b):
    """Chance that a random Kurdish example scores higher than a random other example."""
    return float(np.mean([(x > y) + 0.5 * (x == y) for x in a for y in b]))


def heldout_scores(XA, XB, groupsA, groupsB, n_splits=50):
    """Build the direction on half the data, test how well it separates the other half."""
    ds, aucs = [], []
    groups = np.unique(np.concatenate([groupsA, groupsB]))
    for _ in range(n_splits):
        train = rng.choice(groups, size=len(groups) // 2, replace=False)
        trA, teA = np.isin(groupsA, train), ~np.isin(groupsA, train)
        trB, teB = np.isin(groupsB, train), ~np.isin(groupsB, train)
        direction = XA[trA].mean(0) - XB[trB].mean(0)
        direction /= np.linalg.norm(direction)
        pa, pb = XA[teA] @ direction, XB[teB] @ direction
        ds.append(cohens_d(pa, pb))
        aucs.append(auc(pa, pb))
    return float(np.mean(ds)), float(np.mean(aucs))


def welch_t(a, b):
    return (a.mean(0) - b.mean(0)) / np.sqrt(a.var(0, ddof=1) / len(a) + b.var(0, ddof=1) / len(b) + 1e-6)


def main():
    model = load_model()
    A = S.english_kurdish()
    B = S.english_other()
    sets = {
        "english_kurdish": [s for _, s in A],
        "english_other": [s for _, s in B],
        "kurmanji": S.KURMANJI,
        "other_foreign": S.OTHER_FOREIGN,
        "test_kurdish": S.TEST_KURDISH,
        "test_other": S.TEST_OTHER,
    }
    R, N, T = {}, {}, {}
    for name, sents in sets.items():
        R[name], N[name], T[name] = record(model, sents)
        print(f"recorded {name}: {len(sents)} sentences")

    gA = np.array([g for g, _ in A])
    gB = np.array([g for g, _ in B])
    gC = np.arange(len(S.KURMANJI))
    gD = np.arange(len(S.OTHER_FOREIGN))  # random pairing of halves for the language sets

    # ---------- layer-by-layer: how clear is the Kurdish direction? ----------
    layers = []
    dir_en, dir_ku = [], []
    for L in range(N_LAYERS + 1):
        XA, XB = R["english_kurdish"][:, L], R["english_other"][:, L]
        XC, XD = R["kurmanji"][:, L], R["other_foreign"][:, L]
        d_en, auc_en = heldout_scores(XA, XB, gA, gB)
        d_ku, auc_ku = heldout_scores(XC, XD, gC, gD)
        full_en = XA.mean(0) - XB.mean(0)
        full_ku = XC.mean(0) - XD.mean(0)
        dir_en.append(full_en)
        dir_ku.append(full_ku)
        u_en = full_en / np.linalg.norm(full_en)
        u_ku = full_ku / np.linalg.norm(full_ku)
        # held-out sentences with totally different wording
        pt_k = R["test_kurdish"][:, L] @ u_en
        pt_o = R["test_other"][:, L] @ u_en
        # cross test: does the ENGLISH "Kurdish topic" direction also light up for Kurmanji text?
        pc = XC @ u_en
        pdd = XD @ u_en
        typical_norm = float(np.linalg.norm(np.concatenate([XA, XB]), axis=1).mean())
        layers.append(
            dict(
                layer=L,
                label=("after last layer" if L == N_LAYERS else f"before layer {L}"),
                english_heldout_d=d_en,
                english_heldout_auc=auc_en,
                kurmanji_heldout_d=d_ku,
                kurmanji_heldout_auc=auc_ku,
                newsentences_d=cohens_d(pt_k, pt_o),
                newsentences_auc=auc(pt_k, pt_o),
                cross_english_dir_on_kurmanji_d=cohens_d(pc, pdd),
                cross_english_dir_on_kurmanji_auc=auc(pc, pdd),
                cosine_english_vs_kurmanji_dir=float(u_en @ u_ku),
                direction_size_vs_typical=float(np.linalg.norm(full_en) / typical_norm),
            )
        )
        print(
            f"L{L:2d}  english d={d_en:5.2f} auc={auc_en:.2f} | kurmanji d={d_ku:5.2f} auc={auc_ku:.2f}"
            f" | new-sent d={layers[-1]['newsentences_d']:5.2f} | cross d={layers[-1]['cross_english_dir_on_kurmanji_d']:5.2f}"
            f" | cos={layers[-1]['cosine_english_vs_kurmanji_dir']:.2f}"
        )
    json.dump(layers, open(os.path.join(RESULTS, "layer_scores.json"), "w"), indent=2)
    np.savez(os.path.join(RESULTS, "directions.npz"), english=np.array(dir_en), kurmanji=np.array(dir_ku))

    # ---------- neurons ----------
    def top_neurons(pos, neg, k=15):
        t = welch_t(N[pos], N[neg])  # layer x neuron
        flat = np.argsort(t.ravel())[::-1][:k]
        out = []
        for f in flat:
            L, n = divmod(int(f), t.shape[1])
            # which words make this neuron fire most, across all sentence sets?
            hits = []
            for name, items in T.items():
                for toks, acts in items:
                    for p, tok in enumerate(toks):
                        hits.append((float(acts[L, p, n]), tok, name))
            hits.sort(key=lambda h: -h[0])
            seen, best = set(), []
            for a, tok, name in hits:
                if tok not in seen:
                    seen.add(tok)
                    best.append(dict(token=tok, activation=round(a, 2), from_set=name))
                if len(best) == 6:
                    break
            out.append(
                dict(
                    neuron=f"L{L}.N{n}",
                    layer=L,
                    index=n,
                    t_score=round(float(t[L, n]), 2),
                    mean_on_kurdish=round(float(N[pos][:, L, n].mean()), 3),
                    mean_on_other=round(float(N[neg][:, L, n].mean()), 3),
                    heldout_mean_test_kurdish=round(float(N["test_kurdish"][:, L, n].mean()), 3),
                    heldout_mean_test_other=round(float(N["test_other"][:, L, n].mean()), 3),
                    mean_on_kurmanji=round(float(N["kurmanji"][:, L, n].mean()), 3),
                    mean_on_other_foreign=round(float(N["other_foreign"][:, L, n].mean()), 3),
                    top_tokens=best,
                )
            )
        # how many of the 200 strongest Kurdish neurons are in each layer?
        flat200 = np.argsort(t.ravel())[::-1][:200]
        per_layer = np.bincount(flat200 // t.shape[1], minlength=N_LAYERS).tolist()
        return out, per_layer, t

    top_en, per_layer_en, t_en = top_neurons("english_kurdish", "english_other")
    top_ku, per_layer_ku, t_ku = top_neurons("kurmanji", "other_foreign")
    # neurons that react to Kurdish in BOTH ways (topic in English and the language itself)
    both = np.minimum(t_en, t_ku)
    flat = np.argsort(both.ravel())[::-1][:10]
    shared = [
        dict(neuron=f"L{f // 3072}.N{f % 3072}", t_english=round(float(t_en.ravel()[f]), 2),
             t_kurmanji=round(float(t_ku.ravel()[f]), 2))
        for f in map(int, flat)
    ]
    json.dump(
        dict(english_topic=top_en, kurmanji_language=top_ku, per_layer_top200_english=per_layer_en,
             per_layer_top200_kurmanji=per_layer_ku, shared=shared),
        open(os.path.join(RESULTS, "top_neurons.json"), "w"),
        indent=2,
        ensure_ascii=False,
    )

    # ---------- points for a scatter picture at a middle layer ----------
    pts = {}
    for L in (6, 8):
        u_en = dir_en[L] / np.linalg.norm(dir_en[L])
        u_ku = dir_ku[L] / np.linalg.norm(dir_ku[L])
        pts[L] = {name: dict(x=(R[name][:, L] @ u_en).tolist(), y=(R[name][:, L] @ u_ku).tolist())
                  for name in sets}
    json.dump(pts, open(os.path.join(RESULTS, "scatter_points.json"), "w"))

    with open(os.path.join(RESULTS, "step1_summary.txt"), "w") as f:
        f.write("TOP NEURONS: English Kurdish-topic sentences vs other-country sentences\n")
        for x in top_en:
            f.write(f"{x['neuron']:>10} t={x['t_score']:6.2f}  kurdish={x['mean_on_kurdish']:.2f} other={x['mean_on_other']:.2f}"
                    f"  heldout {x['heldout_mean_test_kurdish']:.2f} vs {x['heldout_mean_test_other']:.2f}"
                    f"  fires on: {[t['token'] for t in x['top_tokens']]}\n")
        f.write("\nTOP NEURONS: Kurmanji sentences vs other foreign-language sentences\n")
        for x in top_ku:
            f.write(f"{x['neuron']:>10} t={x['t_score']:6.2f}  kurmanji={x['mean_on_kurdish']:.2f} other={x['mean_on_other']:.2f}"
                    f"  fires on: {[t['token'] for t in x['top_tokens']]}\n")
        f.write("\nNEURONS STRONG FOR BOTH\n")
        for x in shared:
            f.write(f"{x}\n")
        f.write(f"\nper-layer count of top-200 english-topic neurons: {per_layer_en}\n")
        f.write(f"per-layer count of top-200 kurmanji neurons:      {per_layer_ku}\n")
    print(open(os.path.join(RESULTS, "step1_summary.txt")).read())


if __name__ == "__main__":
    main()
