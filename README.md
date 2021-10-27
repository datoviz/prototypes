# Datoviz experiments and prototypes

This repository contains various ongoing experiments and GUI prototypes.

All prototypes are experimental. Use at your own risks!

## Installation instructions

* [Install Datoviz](https://datoviz.org/tutorials/install/#how-to-install-datoviz)
* Make sure the Datoviz demo is working: `python -c "import datoviz; datoviz.demo()"`
* Clone the prototypes repository: `git clone https://github.com/datoviz/prototypes.git dvzproto`
* Install the `develop` branch of [ibllib](https://github.com/int-brain-lab/ibllib)

## IBL prototypes

### Ephys data viewer

* Usage example: `python ephysview.py`
* Description: shows the full raster plot of an ephys session at the top, and a small section of the raw data on the bottom.
* Use: control+click in the top panel to select a particular time, and load (download + on-the-fly decompression) the corresponding chunk of raw data in the bottom panel.

![](images/ephysview.jpg)


### Coverage viewer

* Usage example: `python coverage.py`
* Requires: `coverage.npy`

![](images/coverage.jpg)


### Spike localization viewer

* Usage example: `python spikeloc.py`

![](images/spikeloc.jpg)


## Other prototypes

* **High-res humain brain viewer**: `brain_highres.py`
* **Molecule viewer**: `molecule.py`
