from pathlib import Path

import numpy as np
import numpy.random as nr

from datoviz import canvas, run, colormap


c = canvas(show_fps=True)
panel = c.panel(controller='arcball')
visual = panel.visual('volume')

texshape = np.array([456, 320, 528])
coef = .004
shape = coef * texshape[[1, 0, 2]]

visual.data('pos', np.atleast_2d(-shape / 2), idx=0)
visual.data('pos', np.atleast_2d(+shape / 2), idx=1)

visual.data('texcoords', np.atleast_2d([0, 0, 0]), idx=0)
visual.data('texcoords', np.atleast_2d([1, 1, 1]), idx=1)

visual.data('length', shape)

# HACK
visual.data('colormap', 5 * np.ones(1))

ROOT = Path(__file__).resolve().parent.parent / 'datoviz'
vol = np.fromfile(ROOT / "data/volume/atlas_25.img", dtype=np.uint16)
vol = vol.reshape(texshape)
visual.volume(vol)

run()
