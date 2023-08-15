# FEBID-pillar-analysis
 Images and code for measuring and analyzing the volumes of pillars deposited by FEBID.

## Content

### /analysis
Jupyter notebooks and supporting python code for measuring the volume of each pillar as well as for aggregating the measurements, performing the linear regression analysis, and generating the plots.

### /data
`.xlsx` and `.csv` files comprising the dose information for each pillar as well as the electron beam and material parameters.

### /images
* `./raw` -- unprocessed SEM images of each 3x3 array of pillars.
* `./masks` -- segmentation masks of each 3x3 array of pillars.

### /results
Processed SEM images of each 3x3 array of pillars with volume measured as if each pillar is a cone (`./cones`), cylinder (`./cylinders`), or using the oultine from the segmentation (`./pillars` -- most accurate results and those used for all subsequent analysis).

Output from `/analysis/1_measure.ipynb`.
<!--
* `/cones` -- processed SEM images of each 3x3 array of pillars with volume measured as if each pillar is a cone.
* `/cones` -- processed SEM images of each 3x3 array of pillars with volume measured as if each pillar is a cylinder.
* `/pillars` -- processed SEM images of each 3x3 array of pillars with volume measured using the outline from the segmentation. Most accurate measurement technique, and the technique used in all subsequent analysis.
* `results.csv` -- table of all recorded volume measurements.
-->


## Installation & Usage
Download or clone the repository
```
git clone https://github.com/lanery/FEBID-pillar-analysis.git
```

Both the raw and processed SEM images of the pillars are directly viewable within `/images` and `/results/pillars`, respectively.

To (re-)run the analysis, first install the requirements (ideally into a fresh conda environment).
```
conda create -n febid python=3.10 jupyterlab  # optional but recommended
pip install -r requirements.txt
```
Start a jupyter lab session.
```
jupyter lab
```
Navigate to `/analysis/` and run `1_measurements.ipynb` and/or `2_plots.ipynb`.


## Citation
Forthcoming...
