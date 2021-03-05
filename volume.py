from pathlib import Path

import numpy as np
import numpy.random as nr

from ibllib.atlas import AllenAtlas
ba = AllenAtlas()

from datoviz import canvas, run, colormap


c = canvas(show_fps=True, pick=True)
panel = c.panel(controller='arcball')
visual = panel.visual('volume')

texshape = np.array([456, 320, 528])
coef = .004
shape = coef * texshape[[1, 0, 2]]

visual.data('pos', np.atleast_2d(-shape / 2), idx=0)
visual.data('pos', np.atleast_2d(+shape / 2), idx=1)

# Box shape.
visual.data('length', np.atleast_2d(shape))

# Transfer function.
fun = np.linspace(0, .05, 256).astype(np.float32)
transfer = c.transfer(fun)
visual.texture(transfer)

# X range of the transfer function
visual.data('transferx', np.array([0, .5]))

# 3D texture with the volume
vol = np.load("atlas.npy")
vol = vol.reshape(texshape)
vol *= 100
V = c.volume(vol)
visual.texture(V)

# 3D texture with the labels
volume_label = np.load("volume_label.npy")
volume_label = volume_label.reshape(tuple(texshape) + (4,))
V_label = c.volume(volume_label)
visual.texture(V_label, idx=1)


@c.connect
def on_mouse_click(x, y, button, modifiers=()):
    # ml: right positive, 0-255
    # dv: dorsal positive
    # ap: anterior negative
    ml, dv, ap, _ = c.pick(x, y)
    ml = (ml + .5) / 256.0
    dv = (dv + .5) / 256.0
    ap = (ap + .5) / 256.0
    ap = 1 - ap  # anterior positive

    nml = texshape[0]
    nap = texshape[2]
    ndv = texshape[1]

    iml = int(round(ml * nml))
    iap = int(round(ap * nap))
    idv = int(round(dv * ndv))

    # ba.label.shape = (nap, nml, ndv)
    l = ba.label[iap, iml, idv]
    print(ba.regions.name[l])

# GUI
gui = c.gui("GUI")

# Transfer function slider
@gui.control("slider_float2", "transferx", vmin=0, vmax=1, force_increasing=True)
def on_change(x0, x1):
    visual.data('transferx', np.array([x0, x1]))

# Clipping
clip = np.zeros(4, dtype=np.float32)
clip[2] = +1

@gui.control("slider_float", "clip offset", vmin=0, vmax=1, value=0)
def on_change(x):
    clip[3] = -x
    visual.data('clip', clip)

run()
