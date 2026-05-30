from __future__ import annotations
import os
import numpy as np

def multi_scale_plot(compiled, source_path: str):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception as e:
        return None
    macros = []
    for sid, sub in compiled.runtime.substrates.items():
        if getattr(sub, 'projector', None) is not None and len(getattr(sub, 'projector_member_ids', [])) > 0:
            macros.append(sub)
    if not macros:
        return None
    out_dir = os.path.join(os.path.dirname(source_path), '_plots')
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(source_path))[0]
    saved = []
    for macro in macros:
        members = [compiled.runtime.substrates[mid] for mid in macro.projector_member_ids if mid in compiled.runtime.substrates]
        if not members:
            continue
        n_members = len(members)
        fig, axes = plt.subplots(n_members + 1, 1, figsize=(8, 2 * (n_members + 1)), sharex=True)
        for i, sub in enumerate(members):
            ax = axes[i]
            if sub.density_traj is None or sub.density_traj.size == 0:
                ax.text(0.5, 0.5, '(no trajectory)', ha='center', transform=ax.transAxes)
                continue
            ax.imshow(sub.density_traj, aspect='auto', origin='lower', extent=[sub.t_traj[0], sub.t_traj[-1], -sub.params.L / 2, sub.params.L / 2], cmap='viridis')
            ax.set_ylabel(sub.name)
            ax.set_title(f'|psi|²(x,t)  —  {sub.name}')
        ax = axes[-1]
        if macro.macro_slow_state_t and macro.macro_slow_state_t_grid:
            ax.plot(macro.macro_slow_state_t_grid, macro.macro_slow_state_t)
            ax.set_xlabel('t')
            ax.set_ylabel(f'{macro.name}.slow_state')
            ax.set_title(f'macro {macro.name} — projected slow variable')
        out_path = os.path.join(out_dir, f'{base}_multiscale_{macro.name}.png')
        fig.tight_layout()
        fig.savefig(out_path, dpi=110)
        plt.close(fig)
        saved.append(out_path)
    return saved

def bravais_3d_render(sub, threshold_frac: float=0.3, out_path: str='bravais.png'):
    if not hasattr(sub, 'D') or sub.D != 3:
        return None
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
    except Exception:
        return None
    rho = np.abs(sub.psi) ** 2
    thresh = threshold_frac * float(rho.max())
    pts = np.argwhere(rho > thresh)
    if pts.size == 0:
        return None
    xs = sub.x
    px = xs[pts[:, 0]]
    py = xs[pts[:, 1]]
    pz = xs[pts[:, 2]]
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(px, py, pz, c=rho[pts[:, 0], pts[:, 1], pts[:, 2]], cmap='plasma', s=8)
    try:
        from runtime.codec_3d import bravais_family
        family = bravais_family(sub.psi, sub.dx)
        ax.set_title(f'{sub.name} — Bravais family: {family}  (threshold {threshold_frac} of peak)')
    except Exception:
        ax.set_title(f'{sub.name} — Bravais detection unavailable')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path