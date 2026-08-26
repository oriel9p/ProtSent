"""H2 test: does pooled-cosine pretraining homogenize per-residue embeddings?

MaxSim scores sum_i max_j A_i.B_j -- it needs residues within a protein to be individually
distinguishable. Mean-pool cosine training spreads gradient 1/L over every residue and rewards
them all pointing at one protein-level direction, which would compress exactly that diversity.

Two measures per sequence, on the residue matrix E (L x d), rows L2-normalized:
  intra_cos  = mean off-diagonal cosine between residues of the SAME protein. Higher = collapsed.
  eff_rank   = participation ratio of E's singular values, exp(entropy of s^2 distribution).
               Scale-free, in [1, d]. Lower = collapsed.

H2 predicts ProtSent-V2 > ESM2 on intra_cos and < on eff_rank, at matched size.
"""
import sys, numpy as np, torch
sys.path.insert(0, "/opt/hpc/ddofer/ProtSent")
import late_interaction as li


DEVICE = "cuda:3"
N_SEQ = 300
MODELS = [
    ("ESM2-35M",         "facebook/esm2_t12_35M_UR50D"),
    ("ProtSent-V2-35M",  "GrimSqueaker/ProtSent-V2-35M"),
    ("ESM2-150M",        "facebook/esm2_t30_150M_UR50D"),
    ("ProtSent-V2-150M", "GrimSqueaker/ProtSent-V2-150M"),
]

seqs_all, _ = li.load_scope40()
seqs = [s for s in seqs_all if 60 <= len(s) <= 400][:N_SEQ]
print(f"{len(seqs)} sequences, mean length {np.mean([len(s) for s in seqs]):.0f}", flush=True)

print(f"\n{'model':<20} {'dim':>5} {'intra_cos':>10} {'eff_rank':>9} {'eff_rank/dim':>13}")
print("-" * 62)
for name, path in MODELS:
    mve, _ = li.build_multivector_encoder(path, proj_dim=0, max_seq_length=512, device=DEVICE)
    mve.eval()
    ic, er = [], []
    with torch.no_grad():
        for i in range(0, len(seqs), 32):
            embs = mve.encode_document(seqs[i:i+32], convert_to_numpy=False,
                                       show_progress_bar=False)
            for e in embs:
                E = torch.nn.functional.normalize(e.float(), dim=-1)
                L = E.shape[0]
                if L < 8:
                    continue
                G = E @ E.T
                # off-diagonal mean: subtract the L ones on the diagonal
                ic.append(((G.sum() - L) / (L * (L - 1))).item())
                s = torch.linalg.svdvals(E) ** 2
                p = s / s.sum()
                er.append(torch.exp(-(p * torch.log(p + 1e-12)).sum()).item())
    d = mve.get_embedding_dimension()
    print(f"{name:<20} {d:>5} {np.mean(ic):>10.4f} {np.mean(er):>9.2f} {np.mean(er)/d:>13.4f}",
          flush=True)
    del mve
    torch.cuda.empty_cache()
