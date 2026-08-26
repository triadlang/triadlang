
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from runtime.core.solver import integrate
from runtime.physics.observables import crystallinity, dominant_wavenumber, ipr, participation_ratio
from stdlib.regimes import resolve_regime

REGIMES = ["B0", "anti_collapse", "HodgkinHuxley", "ENSO_recharge"]
SEED = 0
T = 10.0
N = 128

def freeze():
    baselines = {}
    for name in REGIMES:
        p = resolve_regime(name, seed=SEED, N=N)
        p.T = T
        p.seed = SEED
        result = integrate(p, record_y=True)
        psi = result["psi_final"]
        dx = result["dx"]
        baselines[name] = {
            "crystallinity": float(crystallinity(psi, dx)),
            "k_star": float(dominant_wavenumber(psi, dx)),
            "ipr": float(ipr(psi, dx)),
            "participation": float(participation_ratio(psi, dx)),
        }
        print("  %s: C=%.4f k*=%.4f ipr=%.6f PR=%.2f" % (
            name,
            baselines[name]["crystallinity"],
            baselines[name]["k_star"],
            baselines[name]["ipr"],
            baselines[name]["participation"],
        ))
    return baselines

if __name__ == "__main__":
    fixtures_dir = os.path.join(os.path.dirname(__file__), "fixtures")
    os.makedirs(fixtures_dir, exist_ok=True)
    out_path = os.path.join(fixtures_dir, "inference_baselines.json")

    print("Freezing baselines (T=%.1f, N=%d, seed=%d)..." % (T, N, SEED))
    baselines = freeze()

    print("\nReproducibility check...")
    baselines2 = freeze()

    all_match = True
    for name in REGIMES:
        for k in baselines[name]:
            v1 = baselines[name][k]
            v2 = baselines2[name][k]
            match = abs(v1 - v2) < 1e-10
            if not match:
                print("  MISMATCH %s.%s: %.10f vs %.10f" % (name, k, v1, v2))
                all_match = False

    if all_match:
        print("All baselines reproducible to 10 decimal places.")
    else:
        print("WARNING: baselines not reproducible (stochastic mode?).")
        sys.exit(1)

    with open(out_path, "w") as f:
        json.dump(baselines, f, indent=2)
    print("\nSaved to %s" % out_path)
