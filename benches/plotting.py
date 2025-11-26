import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# def plot_runtime_vs_n(df, outdir):
#     for task in sorted(df['task'].dropna().unique()):
#         sub = df[df['task']==task]
#         if sub.empty: 
#             continue
#         plt.figure()
#         for backend in sorted(sub['backend'].dropna().unique()):
#             ss = sub[sub['backend']==backend].groupby(['nqubits'], as_index=False)['total_s'].median()
#             plt.plot(ss['nqubits'], ss['total_s'], marker='o', label=backend)
#         plt.xlabel('nqubits')
#         plt.ylabel('total time (s)')
#         plt.title(f'Runtime vs nqubits — {task}')
#         plt.legend()
#         plt.tight_layout()
#         plt.savefig(outdir / f'rt_vs_n_{task}.png', dpi=160)
#         plt.close()

def plot_runtime_vs_n(df, outdir):
    for task in sorted(df['task'].dropna().unique()):
        sub = df[df['task'] == task]
        if sub.empty:
            continue
        plt.figure()

        # cpu
        cpu_sub = sub[sub['backend'] == 'cpu']
        if not cpu_sub.empty:
            cpu_med = (
                cpu_sub
                .groupby('nqubits', as_index=False)['total_s']
                .median()
                .sort_values('nqubits')
            )
            plt.plot(cpu_med['nqubits'], cpu_med['total_s'],
                     marker='o', label='cpu')

        # gpu combo
        gpu_sub = sub[sub['backend'] == 'gpu']
        if not gpu_sub.empty:
            for transp in sorted(gpu_sub['transpiler'].dropna().unique()):
                grp = gpu_sub[gpu_sub['transpiler'] == transp]
                if grp.empty:
                    continue
                gpu_med = (
                    grp
                    .groupby('nqubits', as_index=False)['total_s']
                    .median()
                    .sort_values('nqubits')
                )
                label = f"gpu + {transp}"
                plt.plot(gpu_med['nqubits'], gpu_med['total_s'],
                            marker='o', label=label)

        plt.xlabel('nqubits')
        plt.ylabel('total time (s)')
        plt.title(f'Runtime vs nqubits — {task}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / f'rt_vs_n_{task}.png', dpi=160)
        plt.close()


def plot_transpile_sim_breakdown(df, outdir):
    sub = df.groupby(['backend','task'], as_index=False)[['transpile_s','simulate_s']].median()
    plt.figure()
    x = np.arange(len(sub))
    plt.bar(x, sub['transpile_s'], label='transpile')
    plt.bar(x, sub['simulate_s'], bottom=sub['transpile_s'], label='simulate')
    plt.xticks(x, [f"{b}\n{t}" for b,t in zip(sub['backend'], sub['task'])])
    plt.ylabel('time (s)')
    plt.title('Median time breakdown')
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / 'breakdown.png', dpi=160)
    plt.close()

def plot_speedup(df, outdir):
    # pivot by (task, nqubits); need both cpu and gpu rows
    key = ['task','nqubits']
    cpu = df[df['backend']=='cpu'].groupby(key)['total_s'].median()
    gpu = df[df['backend']=='gpu'].groupby(key)['total_s'].median()
    common = cpu.index.intersection(gpu.index)
    if len(common)==0: 
        return
    sp = (cpu[common] / gpu[common]).reset_index(name='speedup')
    plt.figure()
    for task in sorted(sp['task'].unique()):
        ss = sp[sp['task']==task]
        plt.plot(ss['nqubits'], ss['speedup'], marker='o', label=task)
    plt.axhline(1.0, linestyle='--')
    plt.xlabel('nqubits')
    plt.ylabel('CPU/GPU speedup (x)')
    plt.title('Speedup vs nqubits')
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / 'speedup.png', dpi=160)
    plt.close()

def plot_correctness(df, outdir):
    # statevector overlap and sampling TVD
    sv = df[df['overlap'].notna() & (df['overlap']!="")].copy()
    if not sv.empty:
        sv['overlap'] = sv['overlap'].astype(float)
        plt.figure()
        plt.scatter(sv['nqubits'], sv['overlap'], s=12)
        plt.xlabel('nqubits'); plt.ylabel('|<cpu|gpu>|')
        plt.title('Statevector overlap vs nqubits (GPU vs CPU)')
        plt.tight_layout(); plt.savefig(outdir/'overlap.png', dpi=160); plt.close()
    sm = df[df['tvd'].notna() & (df['tvd']!="")].copy()
    if not sm.empty:
        sm['tvd'] = sm['tvd'].astype(float)
        plt.figure()
        plt.scatter(sm['nqubits'], sm['tvd'], s=12)
        plt.xlabel('nqubits'); plt.ylabel('TVD')
        plt.title('Sampling TVD vs nqubits (GPU vs CPU)')
        plt.tight_layout(); plt.savefig(outdir/'tvd.png', dpi=160); plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default=str(PROJECT_ROOT / 'results' / 'result.csv'))
    ap.add_argument('--outdir', default=str(PROJECT_ROOT / 'results'))
    args = ap.parse_args()

    csv_path = Path(args.csv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Loading CSV: {csv_path}")
    print(f"Saving plots to: {outdir}")

    df = pd.read_csv(csv_path)
    plot_runtime_vs_n(df, outdir)
    plot_transpile_sim_breakdown(df, outdir)
    plot_speedup(df, outdir)
    plot_correctness(df, outdir)


if __name__ == '__main__':
    main()
