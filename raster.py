"""
# Raster plot

"""

from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent / "datoviz"

import numpy as np
import numpy.random as nr

from datoviz import canvas, run, colormap

c = canvas(show_fps=True)

panel = c.scene().panel(controller='axes')
visual = panel.visual('point')

amps = np.load(ROOT / "data/misc/spikes.amps.npy")
spike_clusters = np.load(ROOT / "data/misc/spikes.clusters.npy")
spike_depths = np.load(ROOT / "data/misc/spikes.depths.npy")
spike_times = np.load(ROOT / "data/misc/spikes.times.npy")

N = len(spike_times)
print(f"{N} spikes")
pos = np.c_[spike_times, spike_depths, np.zeros(N)]
color = colormap(spike_clusters.astype(np.float64), cmap='glasbey', alpha=.5)

visual.data('pos', pos)
visual.data('color', color)
visual.data('ms', np.array([2.]))

run()
