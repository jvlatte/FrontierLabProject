import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

"""
Module for plotting benchmark results from CSV logs.
"""

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def plot_transpile_and_sim_vs_n(df, outdir):
    required = {'nqubits', 'mode', 'task', 'transpile_s', 'simulate_s', 'transpiler'}
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

        # Filter out gpu + custom with aer backend (keep only custom backend for gpu + custom)
        sub = sub.copy()
        mask = (sub['mode'] == 'gpu') & (sub['transpiler'] == 'custom') & (sub['backend'] == 'aer')
        sub = sub[~mask]

        # median aggregation
        med = (
            sub.groupby(['mode', 'transpiler', 'nqubits'], as_index=False)[['transpile_s', 'simulate_s']]
               .median()
               .sort_values('nqubits')
        )
        if med.empty:
            continue

        # transpilation time plot
        plt.figure()
        for mode, transp in combos:
            line = med[(med['mode'] == mode) & (med['transpiler'] == transp)]
            if line.empty:
                continue
            plt.plot(line['nqubits'], line['transpile_s'], marker='o',
                     label=f"{mode} + {transp}")
        plt.xlabel('nqubits')
        plt.ylabel('transpile time (s)')
        plt.title(f'Transpile time vs nqubits — {task}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / f'transpile_vs_n_{task}.png', dpi=160)
        plt.close()

        # simulation time plot
        plt.figure()
        for mode, transp in combos:
            line = med[(med['mode'] == mode) & (med['transpiler'] == transp)]
            if line.empty:
                continue
            plt.plot(line['nqubits'], line['simulate_s'], marker='o',
                     label=f"{mode} + {transp}")
        plt.xlabel('nqubits')
        plt.ylabel('simulate time (s)')
        plt.title(f'Simulate time vs nqubits — {task}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / f'simulate_vs_n_{task}.png', dpi=160)
        plt.close()


def plot_runtime_vs_n(df, outdir, include_cpu=True):
    has_transpiler = 'transpiler' in df.columns
    # Use 'circuit' column if available, otherwise fall back to 'task'
    group_col = 'circuit' if 'circuit' in df.columns else 'task'

    for group in sorted(df[group_col].dropna().unique()):
        sub = df[df[group_col] == group]
        if sub.empty:
            continue
        plt.figure()

        if include_cpu:
            cpu_sub = sub[sub['mode'] == 'cpu']
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
        gpu_sub = sub[sub['mode'] == 'gpu']
        if not gpu_sub.empty:
            for transp in sorted(gpu_sub['transpiler'].dropna().unique()):
                grp = gpu_sub[gpu_sub['transpiler'] == transp]
                if grp.empty:
                    continue
                if transp == 'custom' and 'backend' in grp.columns:
                    # Only custom backend rows
                    grp = grp[grp['backend'] == 'custom']
                    if grp.empty:
                        continue
                    # --- Pick best median across nL for each nqubits ---
                    if 'nL' in grp.columns and grp['nL'].notna().any():
                        # median per (nL, nqubits)
                        med_by_nL = (
                            grp.groupby(['nL', 'nqubits'], as_index=False)['total_s']
                               .median()
                        )

                        # Find best nL for each nqubits (min total_s)
                        idx_best = med_by_nL.groupby('nqubits')['total_s'].idxmin()
                        best_rows = med_by_nL.loc[idx_best].sort_values('nqubits')

                        label = "gpu + custom (best nL)"
                        line, = plt.plot(best_rows['nqubits'], best_rows['total_s'],
                                 marker='o', label=label)
                        
                        # Annotate each point with the best nL value
                        for _, row in best_rows.iterrows():
                            plt.annotate(f"nL={int(row['nL'])}",
                                         xy=(row['nqubits'], row['total_s']),
                                         xytext=(5, 5),
                                         textcoords='offset points',
                                         fontsize=8,
                                         color=line.get_color())
                        continue  # done with custom
                # default behavior for baseline or other transpilers:
                gpu_med = (
                    grp.groupby('nqubits', as_index=False)['total_s']
                       .median()
                       .sort_values('nqubits')
                )
                label = f"gpu + {transp}"
                plt.plot(gpu_med['nqubits'], gpu_med['total_s'],
                         marker='o', label=label)

        plt.xlabel('nqubits', fontsize=20)
        plt.ylabel('total time (s)', fontsize=20)
        plt.title(f'Runtime vs nqubits — {group}' + ('' if include_cpu else ' (GPU only)'), fontsize=20)
        plt.legend(fontsize=15)
        plt.tight_layout()
        suffix = f'rt_vs_n_{group}.png' if include_cpu else f'rt_vs_n_gpu_{group}.png'
        plt.savefig(outdir / suffix, dpi=160)
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
    Plot memory usage vs nqubits for each task.
    Creates two separate graphs:
      1) RSS memory usage (rss_mb) for all modes/transpilers
      2) GPU memory usage (gpu_mem_mb) for GPU modes/transpilers
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

        # === Graph 1: RSS Memory Usage ===
        if has_rss:
            plt.figure()
            plotted_any = False

            # CPU RSS
            cpu_sub = sub[(sub['mode'] == 'cpu') & sub['rss_mb'].notna()]
            if not cpu_sub.empty:
                if has_transpiler:
                    for transp in sorted(cpu_sub['transpiler'].dropna().unique()):
                        grp = cpu_sub[cpu_sub['transpiler'] == transp]
                        if grp.empty:
                            continue
                        cpu_med = (
                            grp
                            .groupby('nqubits', as_index=False)['rss_mb']
                            .median()
                            .sort_values('nqubits')
                        )
                        if cpu_med.empty:
                            continue
                        plt.plot(cpu_med['nqubits'], cpu_med['rss_mb'],
                                 marker='o', label=f'cpu + {transp}')
                        plotted_any = True
                else:
                    cpu_med = (
                        cpu_sub
                        .groupby('nqubits', as_index=False)['rss_mb']
                        .median()
                        .sort_values('nqubits')
                    )
                    if not cpu_med.empty:
                        plt.plot(cpu_med['nqubits'], cpu_med['rss_mb'],
                                 marker='o', label='cpu')
                        plotted_any = True

            # GPU RSS
            gpu_sub = sub[(sub['mode'] == 'gpu') & sub['rss_mb'].notna()]
            if not gpu_sub.empty:
                if has_transpiler:
                    for transp in sorted(gpu_sub['transpiler'].dropna().unique()):
                        grp = gpu_sub[gpu_sub['transpiler'] == transp]
                        if grp.empty:
                            continue
                        gpu_med = (
                            grp
                            .groupby('nqubits', as_index=False)['rss_mb']
                            .median()
                            .sort_values('nqubits')
                        )
                        if gpu_med.empty:
                            continue
                        plt.plot(gpu_med['nqubits'], gpu_med['rss_mb'],
                                 marker='o', label=f'gpu + {transp}')
                        plotted_any = True
                else:
                    gpu_med = (
                        gpu_sub
                        .groupby('nqubits', as_index=False)['rss_mb']
                        .median()
                        .sort_values('nqubits')
                    )
                    if not gpu_med.empty:
                        plt.plot(gpu_med['nqubits'], gpu_med['rss_mb'],
                                 marker='o', label='gpu')
                        plotted_any = True

            if plotted_any:
                plt.xlabel('nqubits')
                plt.ylabel('RSS Memory (MB)')
                plt.title(f'RSS Memory vs nqubits — {task}')
                plt.legend()
                plt.tight_layout()
                plt.savefig(outdir / f'rss_mem_vs_n_{task}.png', dpi=160)
            plt.close()

        # === Graph 2: GPU Memory Usage ===
        if has_gpu_mem:
            plt.figure()
            plotted_any = False

            gpu_sub = sub[(sub['mode'] == 'gpu') & sub['gpu_mem_mb'].notna()]
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
                        plt.plot(gpu_med['nqubits'], gpu_med['gpu_mem_mb'],
                                 marker='o', label=f'gpu + {transp}')
                        plotted_any = True
                else:
                    gpu_med = (
                        gpu_sub
                        .groupby('nqubits', as_index=False)['gpu_mem_mb']
                        .median()
                        .sort_values('nqubits')
                    )
                    if not gpu_med.empty:
                        plt.plot(gpu_med['nqubits'], gpu_med['gpu_mem_mb'],
                                 marker='o', label='gpu')
                        plotted_any = True

            if plotted_any:
                plt.xlabel('nqubits')
                plt.ylabel('GPU Memory (MB)')
                plt.title(f'GPU Memory vs nqubits — {task}')
                plt.legend()
                plt.tight_layout()
                plt.savefig(outdir / f'gpu_mem_vs_n_{task}.png', dpi=160)
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
        cpu_sub = sub[(sub['mode'] == 'cpu') & sub['total_mem_mb'].notna()]
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
        gpu_sub = sub[(sub['mode'] == 'gpu') & sub['total_mem_mb'].notna()]
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

def plot_total_memory_vs_n_gpu_only(df, outdir):
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

        # === GPU ONLY ===
        gpu_sub = sub[(sub['mode'] == 'gpu') & sub['total_mem_mb'].notna()]
        if gpu_sub.empty:
            plt.close()
            continue

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

                plt.plot(
                    gpu_med['nqubits'],
                    gpu_med['total_mem_mb'],
                    marker='o',
                    label=f"gpu + {transp}"
                )
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
                plt.plot(
                    gpu_med['nqubits'],
                    gpu_med['total_mem_mb'],
                    marker='o',
                    label='gpu'
                )
                plotted_any = True

        if not plotted_any:
            plt.close()
            continue

        plt.xlabel('nqubits')
        plt.ylabel('Total memory usage (MB)')
        plt.title(f'Total GPU memory vs nqubits — {task}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / f'total_gpu_mem_vs_n_{task}.png', dpi=160)
        plt.close()


def plot_kernel_launches_comparison(df, outdir):
    """
    Compare kernel launch metrics between gpu+baseline (Aer) and gpu+custom backends.
    
    Creates figures showing:
      1) num_kernel_launches vs nqubits (Aer's sequential launches vs custom's)
      2) num_graph_replays vs nqubits (custom only - shows graph optimization)
      3) Bar chart comparing launches for each circuit type at fixed nqubits
      4) Fusion ratio comparison (gates before/after fusion)
    """
    required_cols = {'num_kernel_launches', 'mode', 'transpiler', 'backend', 'nqubits'}
    if not required_cols.issubset(set(df.columns)):
        print(f"Skipping kernel launch plots - missing columns: {required_cols - set(df.columns)}")
        return

    # Filter to GPU-only rows with kernel launch data
    gpu_df = df[(df['mode'].isin(['gpu', 'gpu_ref'])) & df['num_kernel_launches'].notna()].copy()
    if gpu_df.empty:
        print("No GPU kernel launch data available for plotting")
        return

    # Convert to numeric
    gpu_df['num_kernel_launches'] = pd.to_numeric(gpu_df['num_kernel_launches'], errors='coerce')
    if 'num_graph_replays' in gpu_df.columns:
        gpu_df['num_graph_replays'] = pd.to_numeric(gpu_df['num_graph_replays'], errors='coerce')
    if 'num_fused_ops' in gpu_df.columns:
        gpu_df['num_fused_ops'] = pd.to_numeric(gpu_df['num_fused_ops'], errors='coerce')
    if 'total_gates' in gpu_df.columns:
        gpu_df['total_gates'] = pd.to_numeric(gpu_df['total_gates'], errors='coerce')

    # Use circuit column if available, else task
    group_col = 'circuit' if 'circuit' in gpu_df.columns else 'task'

    # === Plot 1: Kernel Launches vs nqubits (per circuit type) ===
    for group in sorted(gpu_df[group_col].dropna().unique()):
        sub = gpu_df[gpu_df[group_col] == group]
        if sub.empty:
            continue

        plt.figure(figsize=(10, 6))
        plotted_any = False

        # Aer baseline (gpu + baseline + aer)
        aer_sub = sub[(sub['transpiler'] == 'baseline') & (sub['backend'] == 'aer')]
        if not aer_sub.empty:
            aer_med = (
                aer_sub
                .groupby('nqubits', as_index=False)['num_kernel_launches']
                .median()
                .sort_values('nqubits')
            )
            if not aer_med.empty:
                plt.plot(aer_med['nqubits'], aer_med['num_kernel_launches'],
                         marker='o', linewidth=2, markersize=8,
                         label='gpu + baseline (Aer)')
                plotted_any = True

        # Custom backend (gpu + custom + custom)
        custom_sub = sub[(sub['transpiler'] == 'custom') & (sub['backend'] == 'custom')]
        if not custom_sub.empty:

            # --- AVG across all nL (per nqubits)  ---
            ycol = 'num_graph_replays' if 'num_graph_replays' in custom_sub.columns else 'num_kernel_launches'
            avg_all = (custom_sub.groupby('nqubits', as_index=False)[ycol].mean().sort_values('nqubits'))

            if not avg_all.empty and avg_all[ycol].sum() > 0:
                plt.plot(avg_all['nqubits'], avg_all[ycol],
                         marker='s', linewidth=2, markersize=8,
                         label=f'gpu + custom (avg all nL)')
                plotted_any = True


            # --- Pick best nL for each nqubits (min total_s) ---
            if 'nL' in custom_sub.columns and custom_sub['nL'].notna().any() and 'total_s' in custom_sub.columns:
                # median per (nL, nqubits) for total_s to determine best nL
                med_by_nL = (
                    custom_sub.groupby(['nL', 'nqubits'], as_index=False)
                    .agg({'total_s': 'median', 'num_kernel_launches': 'median', 
                          **({'num_graph_replays': 'median'} if 'num_graph_replays' in custom_sub.columns else {})})
                )
                # Find best nL for each nqubits (min total_s)
                idx_best = med_by_nL.groupby('nqubits')['total_s'].idxmin()
                best_rows = med_by_nL.loc[idx_best].sort_values('nqubits')

                # Graph replays for best nL
                if 'num_graph_replays' in best_rows.columns and best_rows['num_graph_replays'].sum() > 0:
                    line2, = plt.plot(best_rows['nqubits'], best_rows['num_graph_replays'],
                             marker='^', linewidth=2, markersize=8,
                             label='gpu + custom (best nL)')
                    plotted_any = True
            else:
                # Graph replays (this is what custom actually does)
                if 'num_graph_replays' in custom_sub.columns:
                    graph_med = (
                        custom_sub
                        .groupby('nqubits', as_index=False)['num_graph_replays']
                        .median()
                        .sort_values('nqubits')
                    )
                    if not graph_med.empty and graph_med['num_graph_replays'].sum() > 0:
                        plt.plot(graph_med['nqubits'], graph_med['num_graph_replays'],
                                 marker='^', linewidth=2, markersize=8,
                                 label='gpu + custom (graphed) - CPU-side kernel launches')
                        plotted_any = True

        if not plotted_any:
            plt.close()
            continue

        plt.xlabel('Number of Qubits', fontsize=22)
        plt.ylabel('Count', fontsize=22)
        plt.title(f'CPU-side Kernel Launches vs nqubits — {group}\n(Aer sequential vs Custom graphed)', fontsize=22)
        plt.legend(fontsize=15)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(outdir / f'kernel_launches_vs_n_{group}.png', dpi=160)
        plt.close()

    # === Plot 2: Bar chart comparison at specific nqubits values ===
    # Pick a few representative nqubits values
    nqubits_values = sorted(gpu_df['nqubits'].dropna().unique())
    if len(nqubits_values) > 0:
        # Take up to 5 representative values (min, 25%, median, 75%, max)
        indices = np.linspace(0, len(nqubits_values) - 1, min(5, len(nqubits_values)), dtype=int)
        sample_nqubits = [nqubits_values[i] for i in indices]

        for group in sorted(gpu_df[group_col].dropna().unique()):
            sub = gpu_df[gpu_df[group_col] == group]
            if sub.empty:
                continue

            # # Prepare data for bar chart
            # aer_launches = []
            # custom_launches = []
            # custom_replays = []
            # labels = []
            # custom_best = []
            # custom_avg = []

            # for nq in sample_nqubits:
            #     nq_sub = sub[sub['nqubits'] == nq]
            #     if nq_sub.empty:
            #         continue

            #     aer_row = nq_sub[(nq_sub['transpiler'] == 'baseline') & (nq_sub['backend'] == 'aer')]
            #     custom_row = nq_sub[(nq_sub['transpiler'] == 'custom') & (nq_sub['backend'] == 'custom')]

            #     aer_val = aer_row['num_kernel_launches'].median() if not aer_row.empty else 0
            #     custom_val = custom_row['num_kernel_launches'].median() if not custom_row.empty else 0
            #     replay_val = custom_row['num_graph_replays'].median() if (not custom_row.empty and 'num_graph_replays' in custom_row.columns) else 0

            #     aer_launches.append(aer_val)
            #     custom_launches.append(custom_val)
            #     custom_replays.append(replay_val)
            #     labels.append(f'n={nq}')
            # Prepare data for bar chart
            aer_launches = []
            custom_best = []      # best nL (min total_s)
            custom_avg = []       # avg across all nL
            labels = []

            for nq in sample_nqubits:
                nq_sub = sub[sub['nqubits'] == nq]
                if nq_sub.empty:
                    continue

                aer_row = nq_sub[(nq_sub['transpiler'] == 'baseline') & (nq_sub['backend'] == 'aer')]
                custom_rows = nq_sub[(nq_sub['transpiler'] == 'custom') & (nq_sub['backend'] == 'custom')]

                # Aer: kernel launches
                aer_val = aer_row['num_kernel_launches'].median() if not aer_row.empty else 0

                # Custom metric: prefer graph replays, else kernel launches
                if not custom_rows.empty:
                    ycol = 'num_graph_replays' if 'num_graph_replays' in custom_rows.columns else 'num_kernel_launches'

                    # --- avg across all nL (mean-of-median per nL, if nL exists) ---
                    if 'nL' in custom_rows.columns and custom_rows['nL'].notna().any():
                        med_per_nL = (
                            custom_rows.groupby(['nL'], as_index=False)[ycol]
                            .median()
                        )
                        avg_val = float(med_per_nL[ycol].mean()) if not med_per_nL.empty else 0
                    else:
                        avg_val = float(custom_rows[ycol].mean())

                    # --- best nL: choose nL that minimizes total_s, then take its median ycol ---
                    if 'nL' in custom_rows.columns and custom_rows['nL'].notna().any() and 'total_s' in custom_rows.columns:
                        med_by_nL = (
                            custom_rows.groupby(['nL'], as_index=False)
                            .agg({ 'total_s': 'median', ycol: 'median' })
                        )
                        if not med_by_nL.empty:
                            best_idx = med_by_nL['total_s'].idxmin()
                            best_val = float(med_by_nL.loc[best_idx, ycol])
                        else:
                            best_val = 0
                    else:
                        # fallback: just use median across all rows
                        best_val = float(custom_rows[ycol].median())
                else:
                    avg_val = 0
                    best_val = 0

                aer_launches.append(aer_val)
                custom_best.append(best_val)
                custom_avg.append(avg_val)
                labels.append(f'n={nq}')

            if not labels:
                continue

            x = np.arange(len(labels))
            width = 0.25

            fig, ax = plt.subplots(figsize=(10, 6))
            # bars1 = ax.bar(x - width, aer_launches, width, label='Aer (CPU-side kernel launches)', color='#1f77b4')
            # bars3 = ax.bar(x + width, custom_replays, width, label='Custom (CPU-side kernel launches)', color='#2ca02c')
            bars1 = ax.bar(x - width, aer_launches, width,
                           label='Aer (kernel launches)', color='#1f77b4')
            bars2 = ax.bar(x, custom_avg, width,
                           label='Custom avg across nL (graph replays)', color='#ff7f0e')
            bars3 = ax.bar(x + width, custom_best, width,
                           label='Custom best nL (graph replays)', color='#2ca02c')


            ax.set_xlabel('Number of Qubits', fontsize=12)
            ax.set_ylabel('Count', fontsize=12)
            ax.set_title(f'Kernel Launch Comparison — {group}', fontsize=14)
            ax.set_xticks(x)
            ax.set_xticklabels(labels)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3, axis='y')

            # Add value labels on bars
            def add_bar_labels(bars):
                for bar in bars:
                    height = bar.get_height()
                    if height > 0:
                        ax.annotate(f'{int(height)}',
                                    xy=(bar.get_x() + bar.get_width() / 2, height),
                                    xytext=(0, 3),
                                    textcoords="offset points",
                                    ha='center', va='bottom', fontsize=8)

            add_bar_labels(bars1)
            add_bar_labels(bars2)
            add_bar_labels(bars3)

            plt.tight_layout()
            plt.savefig(outdir / f'kernel_launches_bar_{group}.png', dpi=160)
            plt.close()

    print(f"Kernel launch comparison plots saved to {outdir}")


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
    # plot_runtime_vs_n(df, outdir, include_cpu=True)
    plot_runtime_vs_n(df, outdir, include_cpu=False)
    plot_correctness(df, outdir)
    plot_memory_vs_n(df, outdir)
    plot_total_memory_vs_n(df, outdir)
    plot_rt_vs_nL(df, outdir)
    plot_transpile_and_sim_vs_n(df, outdir)
    plot_total_memory_vs_n_gpu_only(df, outdir)
    plot_kernel_launches_comparison(df, outdir)

if __name__ == '__main__':
    main()
    
