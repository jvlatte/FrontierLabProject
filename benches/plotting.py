import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def plot_transpile_and_sim_vs_n(df, outdir):
    required = {'nqubits', 'backend', 'task', 'transpile_s', 'simulate_s', 'transpiler'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    combos = [
        ('cpu', 'baseline'),
        ('cpu', 'custom'),
        ('gpu', 'baseline'),
        ('gpu', 'custom'),
    ]

    for task in sorted(df['task'].dropna().unique()):
        sub = df[df['task'] == task]
        if sub.empty:
            continue

        # median aggregation
        med = (
            sub.groupby(['backend', 'transpiler', 'nqubits'], as_index=False)[['transpile_s', 'simulate_s']]
               .median()
               .sort_values('nqubits')
        )
        if med.empty:
            continue

        # transpilation time plot
        plt.figure()
        for backend, transp in combos:
            line = med[(med['backend'] == backend) & (med['transpiler'] == transp)]
            if line.empty:
                continue
            plt.plot(line['nqubits'], line['transpile_s'], marker='o',
                     label=f"{backend} + {transp}")
        plt.xlabel('nqubits')
        plt.ylabel('transpile time (s)')
        plt.title(f'Transpile time vs nqubits — {task}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / f'transpile_vs_n_{task}.png', dpi=160)
        plt.close()

        # simulation time plot
        plt.figure()
        for backend, transp in combos:
            line = med[(med['backend'] == backend) & (med['transpiler'] == transp)]
            if line.empty:
                continue
            plt.plot(line['nqubits'], line['simulate_s'], marker='o',
                     label=f"{backend} + {transp}")
        plt.xlabel('nqubits')
        plt.ylabel('simulate time (s)')
        plt.title(f'Simulate time vs nqubits — {task}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / f'simulate_vs_n_{task}.png', dpi=160)
        plt.close()


def plot_runtime_vs_n(df, outdir):
    has_transpiler = 'transpiler' in df.columns

    for task in sorted(df['task'].dropna().unique()):
        sub = df[df['task'] == task]
        if sub.empty:
            continue
        plt.figure()
            
        cpu_sub = sub[sub['backend'] == 'cpu']
        if not cpu_sub.empty:
            if has_transpiler:
                for transp in sorted(cpu_sub['transpiler'].dropna().unique()):
                    grp = cpu_sub[cpu_sub['transpiler'] == transp]
                    if grp.empty:
                        continue
                    cpu_med = (
                        grp
                        .groupby('nqubits', as_index=False)['total_s']
                        .median()
                        .sort_values('nqubits')
                    )
                    if cpu_med.empty:
                        continue
                    label = f"cpu + {transp}"
                    plt.plot(cpu_med['nqubits'], cpu_med['total_s'],
                             marker='o', label=label)
            else:
                # fallback: single cpu line
                cpu_med = (
                    cpu_sub
                    .groupby('nqubits', as_index=False)['total_s']
                    .median()
                    .sort_values('nqubits')
                )
                if not cpu_med.empty:
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
    sub = df.copy()

    if 'transpiler' in sub.columns:
        tr = sub['transpiler'].fillna('')
        sub['backend_label'] = np.where(
            tr != '',
            sub['backend'] + '+' + tr,
            sub['backend']
        )
    else:
        sub['backend_label'] = sub['backend']

    agg = sub.groupby(['backend_label', 'task'], as_index=False)[['transpile_s', 'simulate_s']].median()

    plt.figure()
    x = np.arange(len(agg))
    plt.bar(x, agg['transpile_s'], label='transpile')
    plt.bar(x, agg['simulate_s'], bottom=agg['transpile_s'], label='simulate')
    plt.xticks(x, [f"{b}\n{t}" for b, t in zip(agg['backend_label'], agg['task'])])
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


def plot_memory_vs_n(df, outdir):
    """
    Plot memory usage vs nqubits for each task, with separate curves for:
      - cpu          -> rss_mb
      - gpu + X      -> gpu_mem_mb (split by transpiler: baseline/custom/etc.)
    """
    has_transpiler = 'transpiler' in df.columns
    has_rss = 'rss_mb' in df.columns
    has_gpu_mem = 'gpu_mem_mb' in df.columns

    if not has_rss and not has_gpu_mem:
        return  # nothing to plot

    for task in sorted(df['task'].dropna().unique()):
        sub = df[df['task'] == task]
        if sub.empty:
            continue

        plt.figure()
        plotted_any = False

        # 1) CPU: use rss_mb
        if has_rss:
            cpu_sub = sub[(sub['backend'] == 'cpu') & sub['rss_mb'].notna()]
            if not cpu_sub.empty:
                cpu_med = (
                    cpu_sub
                    .groupby('nqubits', as_index=False)['rss_mb']
                    .median()
                    .sort_values('nqubits')
                )
                plt.plot(cpu_med['nqubits'], cpu_med['rss_mb'],
                         marker='o', label='cpu (RSS)')
                plotted_any = True

        # 2) GPU: use gpu_mem_mb, split by transpiler => gpu + baseline/custom
        if has_gpu_mem:
            gpu_sub = sub[(sub['backend'] == 'gpu') & sub['gpu_mem_mb'].notna()]
            if not gpu_sub.empty:
                if has_transpiler:
                    for transp in sorted(gpu_sub['transpiler'].dropna().unique()):
                        grp = gpu_sub[gpu_sub['transpiler'] == transp]
                        if grp.empty:
                            continue
                        gpu_med = (
                            grp
                            .groupby('nqubits', as_index=False)['gpu_mem_mb']
                            .median()
                            .sort_values('nqubits')
                        )
                        if gpu_med.empty:
                            continue
                        label = f"gpu + {transp} (VRAM)"
                        plt.plot(gpu_med['nqubits'], gpu_med['gpu_mem_mb'],
                                 marker='o', label=label)
                        plotted_any = True
                else:
                    # fallback: single gpu line
                    gpu_med = (
                        gpu_sub
                        .groupby('nqubits', as_index=False)['gpu_mem_mb']
                        .median()
                        .sort_values('nqubits')
                    )
                    if not gpu_med.empty:
                        plt.plot(gpu_med['nqubits'], gpu_med['gpu_mem_mb'],
                                 marker='o', label='gpu (VRAM)')
                        plotted_any = True

        if not plotted_any:
            plt.close()
            continue

        plt.xlabel('nqubits')
        plt.ylabel('Memory usage (MB)')
        plt.title(f'Memory vs nqubits — {task}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / f'mem_vs_n_{task}.png', dpi=160)
        plt.close()


def plot_total_memory_vs_n(df, outdir):
    has_rss = 'rss_mb' in df.columns
    has_gpu_mem = 'gpu_mem_mb' in df.columns
    has_transpiler = 'transpiler' in df.columns

    if not (has_rss or has_gpu_mem):
        return  # nothing to compute

    for task in sorted(df['task'].dropna().unique()):
        sub = df[df['task'] == task]
        if sub.empty:
            continue

        sub = sub.copy()

        # compute total_mem_mb safely
        if has_rss and has_gpu_mem:
            sub['total_mem_mb'] = sub['rss_mb'].fillna(0) + sub['gpu_mem_mb'].fillna(0)
        elif has_rss:
            sub['total_mem_mb'] = sub['rss_mb']
        else:
            sub['total_mem_mb'] = sub['gpu_mem_mb']

        plt.figure()
        plotted_any = False

        # 1) CPU: aggregate all cpu rows (transpiler ignored)
        cpu_sub = sub[(sub['backend'] == 'cpu') & sub['total_mem_mb'].notna()]
        if not cpu_sub.empty:
            cpu_med = (
                cpu_sub
                .groupby('nqubits', as_index=False)['total_mem_mb']
                .median()
                .sort_values('nqubits')
            )
            if not cpu_med.empty:
                plt.plot(cpu_med['nqubits'], cpu_med['total_mem_mb'],
                         marker='o', label='cpu (total mem)')
                plotted_any = True

        # 2) GPU: split by transpiler so you get gpu+baseline / gpu+custom
        gpu_sub = sub[(sub['backend'] == 'gpu') & sub['total_mem_mb'].notna()]
        if not gpu_sub.empty:
            if has_transpiler:
                for transp in sorted(gpu_sub['transpiler'].dropna().unique()):
                    grp = gpu_sub[gpu_sub['transpiler'] == transp]
                    if grp.empty:
                        continue
                    gpu_med = (
                        grp
                        .groupby('nqubits', as_index=False)['total_mem_mb']
                        .median()
                        .sort_values('nqubits')
                    )
                    if gpu_med.empty:
                        continue
                    label = f"gpu + {transp} (total mem)"
                    plt.plot(gpu_med['nqubits'], gpu_med['total_mem_mb'],
                             marker='o', label=label)
                    plotted_any = True
            else:
                # fallback: single gpu line
                gpu_med = (
                    gpu_sub
                    .groupby('nqubits', as_index=False)['total_mem_mb']
                    .median()
                    .sort_values('nqubits')
                )
                if not gpu_med.empty:
                    plt.plot(gpu_med['nqubits'], gpu_med['total_mem_mb'],
                             marker='o', label='gpu (total mem)')
                    plotted_any = True

        if not plotted_any:
            plt.close()
            continue

        plt.xlabel('nqubits')
        plt.ylabel('Total memory usage (MB)')
        plt.title(f'Total memory vs nqubits — {task}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / f'total_mem_vs_n_{task}.png', dpi=160)
        plt.close()

def plot_rt_vs_nL(df, outdir):
    has_nL = 'nL' in df.columns
    if not has_nL:
        return
    for task in sorted(df['task'].dropna().unique()):
        sub = df[df['task'] == task]
        if sub.empty:
            continue
        plt.figure()
        for nL in sorted(sub['nL'].dropna().unique()):
            grp = sub[sub['nL'] == nL]
            if grp.empty:
                continue
            med = (
                grp
                .groupby('nqubits', as_index=False)['total_s']
                .median()
                .sort_values('nqubits')
            )
            if med.empty:
                continue
            plt.plot(med['nqubits'], med['total_s'],
                     marker='o', label=f'nL={nL}')
        plt.xlabel('nqubits')
        plt.ylabel('total time (s)')
        plt.title(f'Runtime vs nqubits — {task}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / f'rt_vs_n_nL_{task}.png', dpi=160)
        plt.close()


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
    plot_memory_vs_n(df, outdir)
    plot_total_memory_vs_n(df, outdir)
    plot_rt_vs_nL(df, outdir)
    plot_transpile_and_sim_vs_n(df, outdir)

if __name__ == '__main__':
    main()
