"""Render the three training losses and mean Q curves from a local W&B run."""
import argparse
import json
from pathlib import Path
import shutil
import tempfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FuncFormatter
import numpy as np
from wandb.proto.wandb_internal_pb2 import Record
from wandb.sdk.internal.datastore import DataStore


def read_history(source):
    # W&B's scanner opens r+b; scan a private copy to keep the source untouched.
    rows = []
    with tempfile.TemporaryDirectory() as directory:
        copy = Path(directory) / 'history.wandb'
        shutil.copyfile(source, copy)
        store = DataStore()
        store.open_for_scan(str(copy))
        try:
            while (data := store.scan_data()) is not None:
                record = Record()
                record.ParseFromString(data)
                if record.HasField('history'):
                    rows.append({item.key or '/'.join(item.nested_key): json.loads(item.value_json)
                                 for item in record.history.item})
        finally:
            store.close()
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-file', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rows = [r for r in read_history(args.run_file) if 'training/critic_loss' in r]
    keys = ['critic_loss', 'edit_actor_loss', 'actor_loss', 'target_q_mean', 'edit_q', 'q']
    x = np.array([r['training/num_updates'] for r in rows])
    values = {k: np.array([r['training/' + k] for r in rows]) for k in keys}
    assert len(x) and np.all(np.diff(x) > 0)
    assert all(np.isfinite(v).all() for v in values.values())
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 28,
        'axes.titlesize': 38, 'axes.titleweight': 'bold', 'axes.labelsize': 30,
        'xtick.labelsize': 26, 'ytick.labelsize': 26, 'legend.fontsize': 25,
        'axes.labelcolor': '#303440', 'text.color': '#252937',
        'xtick.color': '#626877', 'ytick.color': '#626877',
        'axes.spines.top': True, 'axes.spines.right': True,
        'axes.edgecolor': '#AEB7C7', 'axes.linewidth': 1.0,
        'pdf.fonttype': 42, 'svg.fonttype': 'none',
    })

    def canvas(title, ylabel, *, comparison=False):
        fig = plt.figure(figsize=(12.8, 7.2))
        # Loss panels share a compact layout; the comparison reserves a legend row.
        ax = fig.add_axes([.155, .20, .81, .59 if comparison else .65])
        fig.set_facecolor('white')
        ax.set_facecolor('white')
        fig.suptitle(title, x=.5, y=.975, fontsize=34, fontweight='bold')
        ax.set_xlabel('Network update', labelpad=13)
        ax.set_ylabel(ylabel, labelpad=13)
        ax.xaxis.set_label_coords(.5, -.19)
        ax.yaxis.set_label_coords(-.14, .5)
        ax.set_axisbelow(True)
        ax.grid(axis='y', color='#E9EBF2', linewidth=1.0)
        ax.spines['left'].set_visible(True)
        ax.set_xlim(0, max(x) + 1)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'))
        ax.tick_params(axis='both', direction='out', length=5, width=1.0,
                       color='#AEB7C7', pad=8, top=False, right=False)
        return fig, ax

    def save(fig, name):
        for suffix in ('png', 'pdf', 'svg'):
            path = args.output / f'{name}.{suffix}'
            fig.savefig(path, dpi=240, facecolor='white')
            if suffix == 'svg':
                path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines()) + '\n')
        plt.close(fig)

    def endpoint(ax, value, color, label, offset):
        ax.scatter([x[-1]], [value], s=125, color=color, edgecolors='white',
                   linewidths=2, zorder=8)
        ax.annotate(label, xy=(x[-1], value), xytext=offset,
                    textcoords='offset points', ha='right' if offset[0] < 0 else 'left',
                    va='center', fontsize=23, fontweight='bold', color=color,
                    bbox=dict(boxstyle='round,pad=0.30,rounding_size=0.2',
                              facecolor='white', edgecolor=color, linewidth=1.2),
                    arrowprops=dict(arrowstyle='-', color=color, linewidth=1.4,
                                    shrinkA=5, shrinkB=7), zorder=9)

    specs = [('critic_loss', 'Critic Loss', '#3973F6'),
             ('edit_actor_loss', 'Edit Actor Loss', '#F76C50'),
             ('actor_loss', 'PI05 Actor Loss', '#9560DB')]
    for key, title, color in specs:
        fig, ax = canvas(title, 'Loss')
        y = values[key]
        ax.plot(x, y, color=color, linewidth=3.7, solid_capstyle='round', solid_joinstyle='round')
        if key == 'critic_loss':
            # Keep every point while revealing the small late-stage losses.
            ax.set_yscale('log')
            ax.set_ylabel('Loss (log scale)')
            ax.set_yticks([.001, .01, .1], labels=['0.001', '0.01', '0.1'])
            ax.minorticks_off()
            ax.set_ylim(y.min() / 1.35, y.max() * 1.35)
        else:
            padding = .12 * np.ptp(y)
            ax.set_ylim(max(0, y.min() - padding), y.max() + padding)
        endpoint(ax, y[-1], color, f'{y[-1]:.5f}' if key != 'edit_actor_loss'
                 else f'{y[-1]:.3f}', (-18, 38))
        save(fig, key)

    fig, ax = canvas('Q-value Estimates', 'Mean Q', comparison=True)
    for key, label, color, style, marker in [
        ('target_q_mean', 'Target Q', '#3973F6', '-', 'o'),
        ('edit_q', 'Edit Q', '#F76C50', '--', 's'),
        ('q', 'Training Q', '#9560DB', '-.', '^'),
    ]:
        ax.plot(x, values[key], label=label, color=color, linestyle=style,
                linewidth=3.5, marker=marker, markevery=7, markersize=7,
                markeredgecolor='white', markeredgewidth=1.0,
                solid_capstyle='round', dash_capstyle='round')
        offset_y = {'target_q_mean': -60, 'edit_q': -15, 'q': 30}[key]
        endpoint(ax, values[key][-1], color, f'{values[key][-1]:.4f}', (22, offset_y))
    # A narrow right-hand label gutter keeps close final Q values legible.
    ax.set_xlim(0, max(x) + 12)
    ax.set_xticks([0, 10, 20, 30, 40, 50])
    ax.legend(loc='lower center', bbox_to_anchor=(.5, 1.01), ncol=3,
              frameon=False, handlelength=2.0, columnspacing=1.5, handletextpad=.55)
    save(fig, 'q_values')
    manifest = {'source': str(args.run_file), 'points': len(x),
                'x': 'training/num_updates', 'smoothing': None,
                'updates': x.tolist(), 'values': {k: v.tolist() for k, v in values.items()}}
    (args.output / 'plot_data.json').write_text(json.dumps(manifest, indent=2))
    print(f'Wrote four plots (PNG/PDF/SVG), {len(x)} raw points each, to {args.output}')


if __name__ == '__main__':
    main()
