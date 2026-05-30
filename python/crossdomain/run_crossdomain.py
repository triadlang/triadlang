import subprocess, sys, os, re
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

def run_tri(name, src):
    path = os.path.join(HERE, name + '.tri')
    open(path, 'w').write(src)
    out = subprocess.run([sys.executable, 'triadc', path], cwd=ROOT, capture_output=True, text=True, timeout=600).stdout
    res = {}
    for line in out.splitlines():
        m = re.match('\\s*(\\w+)\\s*=\\s*(\\{.*\\})', line)
        if m:
            try:
                res[m.group(1)] = eval(m.group(2))
            except Exception:
                pass
    return res

def decl(entities):
    return ''.join((f'reg {n} : {r} = 4;\n' for n, r in entities))

def obs(entities, metrics='k_star, crystallinity, peak, atom_count'):
    return ''.join((f'OBSERVE {n} {metrics};\n' for n, _ in entities))

def solo(name, regime, T=18.0, metrics='k_star, crystallinity, peak, atom_count'):
    r = run_tri(f'_solo_{name}', f'@T({T})\nreg {name} : {regime} = 4;\nevolve {name} for {T};\nOBSERVE {name} {metrics};\n')
    return r.get(name, {})

def fit(d):
    return d.get('crystallinity', 0) * d.get('atom_count', 0)
line = '=' * 72
print(line)
print('SCENARIO 1 — Neuroscience x Climate  (entrainment?)')
print('  HodgkinHuxley neuron  <->  ENSO_recharge climate, pair-coupled')
ents = [('neuron', 'HodgkinHuxley'), ('climate', 'ENSO_recharge')]
src = f'@T(18.0)\n{decl(ents)}pair(neuron, climate) kappa=-2.5 for T=18.0;\n{obs(ents)}'
c = run_tri('s1_neuro_climate', src)
s_neuron, s_climate = (solo('neuron', 'HodgkinHuxley'), solo('climate', 'ENSO_recharge'))
print(f"  neuron  k*: solo {s_neuron.get('k_star', 0):.3f} -> coupled {c.get('neuron', {}).get('k_star', 0):.3f}")
print(f"  climate k*: solo {s_climate.get('k_star', 0):.3f} -> coupled {c.get('climate', {}).get('k_star', 0):.3f}")
dk_solo = abs(s_neuron.get('k_star', 0) - s_climate.get('k_star', 0))
dk_coup = abs(c.get('neuron', {}).get('k_star', 0) - c.get('climate', {}).get('k_star', 0))
print(f"  |Δk*| solo={dk_solo:.3f}  coupled={dk_coup:.3f}  -> {('ENTRAINMENT (pulled together)' if dk_coup < dk_solo else 'no entrainment')}")
print(line)
print('SCENARIO 2 — Origin of life  (chemistry/biology arena)')
print('  England autopoiesis + Eigen hypercycle + Belousov-Zhabotinsky, ring')
ents = [('autopoiesis', 'England_autopoietic'), ('hypercycle', 'Eigen_hypercycle'), ('bzchem', 'Belousov_Zhabotinsky')]
src = f'@T(18.0)\n{decl(ents)}ring(autopoiesis, hypercycle, bzchem) kappa=-2.5 for T=18.0;\n{obs(ents)}'
c = run_tri('s2_origin_of_life', src)
for n, r in ents:
    d = c.get(n, {})
    print(f"  {n:12s} ({r:20s}) atoms={d.get('atom_count', 0):3d} C={d.get('crystallinity', 0):.3f} fit={fit(d):.2f}")
print(line)
print('SCENARIO 3 — Finance x Materials')
print('  LSV_market volatility  <->  MaxwellWiechert relaxation, pair')
ents = [('market', 'LSV_market'), ('polymer', 'MaxwellWiechert')]
src = f'@T(18.0)\n{decl(ents)}pair(market, polymer) kappa=-2.5 for T=18.0;\n{obs(ents)}'
c = run_tri('s3_finance_materials', src)
for n, r in ents:
    d = c.get(n, {})
    print(f"  {n:8s} ({r:16s}) atoms={d.get('atom_count', 0):3d} C={d.get('crystallinity', 0):.3f} k*={d.get('k_star', 0):.3f}")
print(line)
print('SCENARIO 4 — Cosmology arena')
print('  Cepheid star + DarkMatter halo + Cosmological inflation, ring')
ents = [('star', 'Cepheid_pulsator'), ('halo', 'DarkMatter_halo'), ('inflation', 'Cosmological_inflation')]
src = f'@T(18.0)\n{decl(ents)}ring(star, halo, inflation) kappa=-2.0 for T=18.0;\n{obs(ents)}'
c = run_tri('s4_cosmos', src)
for n, r in ents:
    d = c.get(n, {})
    print(f"  {n:10s} ({r:22s}) atoms={d.get('atom_count', 0):3d} C={d.get('crystallinity', 0):.3f} fit={fit(d):.2f}")
print(line)
print('SCENARIO 5 — GRAND TOURNAMENT (10 domains, one ring)')
tour = [('neuron', 'HodgkinHuxley'), ('climate', 'ENSO_recharge'), ('polymer', 'MaxwellWiechert'), ('star', 'Cepheid_pulsator'), ('market', 'LSV_market'), ('cell', 'England_autopoietic'), ('hypercycle', 'Eigen_hypercycle'), ('bz', 'Belousov_Zhabotinsky'), ('halo', 'DarkMatter_halo'), ('inflation', 'Cosmological_inflation')]
src = f"@T(18.0)\n{decl(tour)}ring({', '.join((n for n, _ in tour))}) kappa=-2.5 for T=18.0;\n{obs(tour)}"
c = run_tri('s5_grand_tournament', src)
board = sorted(((fit(c.get(n, {})), n, r, c.get(n, {})) for n, r in tour), reverse=True)
print(f"  {'rank':>4} {'entity':>10} {'regime':>22} {'atoms':>6} {'C':>6} {'fitness':>8}")
for i, (f, n, r, d) in enumerate(board, 1):
    print(f"  {i:>4} {n:>10} {r:>22} {d.get('atom_count', 0):6d} {d.get('crystallinity', 0):6.3f} {f:8.2f}")
print(f'  >>> CHAMPION: {board[0][1]} ({board[0][2]})')
print(line)
print('SCENARIO 6 — DEPTH TEST: does the matchup matter, or just drive?')
print('  ENSO climate placed in 3 different rings; if its outcome depends on')
print('  WHO the partners are -> strategic depth. If identical -> generic drive.')

def enso_in(partners, label, tag):
    ents = [('climate', 'ENSO_recharge')] + partners
    src = f"@T(18.0)\n{decl(ents)}ring({', '.join((n for n, _ in ents))}) kappa=-2.5 for T=18.0;\nOBSERVE climate k_star, crystallinity, peak, atom_count;\n"
    d = run_tri(f's6_{tag}', src).get('climate', {})
    print(f"  vs {label:28s} -> climate atoms={d.get('atom_count', 0):3d} C={d.get('crystallinity', 0):.3f} k*={d.get('k_star', 0):.3f}")
enso_in([('a', 'HodgkinHuxley'), ('b', 'MaxwellWiechert'), ('c', 'Cepheid_pulsator')], 'heterogeneous [HH,MW,Ceph]', 'het')
enso_in([('a', 'B0'), ('b', 'B0'), ('c', 'B0')], 'neutral [3x B0]', 'b0')
enso_in([('a', 'anti_collapse'), ('b', 'anti_collapse'), ('c', 'anti_collapse')], 'focusing [3x anti_collapse]', 'ac')
print(line)
print('done — scenario .tri files written in crossdomain/')