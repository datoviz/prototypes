import numpy as np
from oneibl.one import ONE
from ibllib.pipes.histology import coverage
from ibllib.atlas import AllenAtlas


# one = ONE()
# trajs = one.alyx.rest(
#     'trajectories', 'list',
#     django=(
#         'provenance,70,'
#         'probe_insertion__session__project__name__icontains,ibl_neuropixel_brainwide_01,'
#         'probe_insertion__session__qc__lt,50,'
#         'probe_insertion__json__extended_qc__alignment_count__gt,0,'
#         'probe_insertion__session__extended_qc__behavior,1'
#     )
# )
# vol = coverage(trajs=trajs, ba=ba)
# vol[np.isnan(vol)] = 0
# np.save('coverage.npy', vol)

# ba = AllenAtlas()

vol = np.load('coverage.npy')
print(vol)
