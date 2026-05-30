import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from runtime.equation_runtime import EquationRuntime
np.random.seed(42)
N = 128
L = 32.0
x = np.linspace(-L / 2, L / 2, N, endpoint=False)

def make_signal(cls, noise=0.1):
    if cls == 0:
        return (np.sin(2 * 2 * np.pi / L * x) + noise * np.random.randn(N)).astype(np.float64)
    elif cls == 1:
        return (np.sin(8 * 2 * np.pi / L * x) + noise * np.random.randn(N)).astype(np.float64)
    else:
        return (np.exp(-x ** 2 / 4.0) + noise * np.random.randn(N)).astype(np.float64)
n_per_class = 40
labels = np.array([c for c in range(3) for _ in range(n_per_class)])
raw_signals = np.array([make_signal(c) for c in labels])
raw_signals = raw_signals / (np.abs(raw_signals).max(axis=1, keepdims=True) + 1e-10)
n = len(labels)
perm = np.random.permutation(n)
ntr = int(0.7 * n)
train_idx, test_idx = (perm[:ntr], perm[ntr:])
X_train_raw, X_test_raw = (raw_signals[train_idx], raw_signals[test_idx])
y_train, y_test = (labels[train_idx], labels[test_idx])
print('=' * 72)
print('  BENCHMARK: Equation Runtime vs Traditional ML')
print('=' * 72)
print(f'  Dataset: {n} samples, 3 classes, N={N}')
print(f'  Train: {ntr}, Test: {n - ntr}')
print()
results = {}
print('  [1/4] Equation Runtime (4 substrates, ring, B0)...')
rt = EquationRuntime(n_substrates=4, regime='B0', N=N, coupling='ring', kappa=-3.0)

def eq_extract(signals):
    states = []
    for sig in signals:
        rt.inject(sig, substrate_idx=0)
        r = rt.run(T=3.0)
        states.append(r['total_state'])
    return np.array(states)
t0 = time.perf_counter()
Xtr_eq = eq_extract(X_train_raw)
Xte_eq = eq_extract(X_test_raw)
eq_feat_time = time.perf_counter() - t0
centroids = np.array([Xtr_eq[y_train == c].mean(0) for c in range(3)])
preds = np.array([np.argmin([np.sum((Xte_eq[i] - centroids[c]) ** 2) for c in range(3)]) for i in range(len(Xte_eq))])
eq_acc = (preds == y_test).mean()
results['equation'] = {'accuracy': eq_acc, 'feat_time': eq_feat_time, 'inference_time': eq_feat_time * (n - ntr) / n, 'memory_kb': rt.memory_kb(), 'params': 0}
print(f'        Accuracy: {eq_acc:.2%}')
print(f'        Feature extraction: {eq_feat_time:.2f}s')
print(f'        Memory: {rt.memory_kb():.1f} KB, Params: 0')
print('  [2/4] sklearn MLPClassifier (256, 128, 64)...')
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(256, 128, 64), max_iter=500, random_state=42)
t0 = time.perf_counter()
mlp.fit(X_train_raw, y_train)
mlp_train = time.perf_counter() - t0
t0 = time.perf_counter()
mlp_pred = mlp.predict(X_test_raw)
mlp_infer = time.perf_counter() - t0
mlp_acc = (mlp_pred == y_test).mean()
n_params = sum((coef.size for coef in mlp.coefs_)) + sum((i.size for i in mlp.intercepts_))
mlp_mem = n_params * 8 / 1024
results['mlp'] = {'accuracy': mlp_acc, 'train_time': mlp_train, 'inference_time': mlp_infer, 'memory_kb': mlp_mem, 'params': n_params}
print(f'        Accuracy: {mlp_acc:.2%}')
print(f'        Train: {mlp_train:.3f}s, Infer: {mlp_infer * 1000:.2f}ms')
print(f'        Memory: {mlp_mem:.1f} KB, Params: {n_params}')
print('  [3/4] sklearn RandomForest (100 trees)...')
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
t0 = time.perf_counter()
rf.fit(X_train_raw, y_train)
rf_train = time.perf_counter() - t0
t0 = time.perf_counter()
rf_pred = rf.predict(X_test_raw)
rf_infer = time.perf_counter() - t0
rf_acc = (rf_pred == y_test).mean()
rf_mem = sum((t.tree_.node_count for t in rf.estimators_)) * 40 / 1024
results['rf'] = {'accuracy': rf_acc, 'train_time': rf_train, 'inference_time': rf_infer, 'memory_kb': rf_mem, 'params': sum((t.tree_.node_count for t in rf.estimators_))}
print(f'        Accuracy: {rf_acc:.2%}')
print(f'        Train: {rf_train:.3f}s, Infer: {rf_infer * 1000:.2f}ms')
print(f'        Memory: {rf_mem:.1f} KB')
print('  [4/4] Ridge on raw signal (linear baseline)...')
try:
    from sklearn.linear_model import RidgeClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    ridge = make_pipeline(StandardScaler(), RidgeClassifier(alpha=1.0))
    t0 = time.perf_counter()
    ridge.fit(X_train_raw, y_train)
    ridge_train = time.perf_counter() - t0
    t0 = time.perf_counter()
    ridge_pred = ridge.predict(X_test_raw)
    ridge_infer = time.perf_counter() - t0
    ridge_acc = (ridge_pred == y_test).mean()
except ImportError:
    ridge_acc = 0
    ridge_train = 0
    ridge_infer = 0
results['ridge'] = {'accuracy': ridge_acc, 'train_time': ridge_train, 'inference_time': ridge_infer, 'memory_kb': N * 8 / 1024, 'params': N}
print(f'        Accuracy: {ridge_acc:.2%}')
print(f'        Train: {ridge_train:.3f}s')
print()
print('=' * 72)
print('  SUMMARY')
print('=' * 72)
print(f"  {'Method':<30} {'Accuracy':>10} {'Memory':>12} {'Params':>10}")
print(f"  {'-' * 30} {'-' * 10} {'-' * 12} {'-' * 10}")
for name, key in [('Equation Runtime', 'equation'), ('MLP (256,128,64)', 'mlp'), ('Random Forest (100)', 'rf'), ('Ridge (linear)', 'ridge')]:
    r = results[key]
    print(f"  {name:<30} {r['accuracy']:>9.1%} {r['memory_kb']:>10.1f}KB {r['params']:>10,}")
print()
print(f'  Equation vs MLP: {mlp_mem / rt.memory_kb():.0f}x less memory, {n_params:,} fewer params')
print(f'  Equation accuracy: {eq_acc:.1%} (zero weights)')
print('=' * 72)