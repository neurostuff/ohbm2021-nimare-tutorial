---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.10.1
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# OHBM 2021 NiMARE tutorial

```python
%matplotlib inline
import json
import os.path as op
from pprint import pprint

import matplotlib.pyplot as plt
from nilearn import plotting

import nimare
```

```python
DATA_DIR = op.abspath("../data/nimare_tutorial/")
```

## Basics of NiMARE datasets
NiMARE relies on a specification for meta-analytic datasets named [NIMADS](https://github.com/neurostuff/NIMADS). NIMADS is currently under development.

```python
with open(op.join(DATA_DIR, "nidm_pain_dset.json"), "r") as fo:
    data = json.load(fo)

reduced_data = {k: v for k, v in data.items() if k in list(data.keys())[:2]}

pprint(reduced_data)
```

```python
dset_dir = nimare.extract.download_nidm_pain(data_dir=DATA_DIR)
pain_dset = nimare.dataset.Dataset(op.join(DATA_DIR, "nidm_pain_dset.json"))
pain_dset.update_path(dset_dir)
```

The `Dataset` stores most relevant information as properties- specifically pandas `DataFrame`s.

```python
pain_dset.coordinates.head()
```

```python
pain_dset.metadata.head()
```

```python
pain_dset.images.head()
```

There are functions to convert common formats for meta-analysis datasets- namely [Neurosynth](https://github.com/neurosynth/neurosynth-data) and [Sleuth](http://brainmap.org/sleuth/) files.

<!-- #region -->
Downloading and converting the Neurosynth dataset takes a long time, so we will use a pregenerated version of the dataset. However, here is the code we would use to download and convert the dataset from scratch:

```python
nimare.extract.fetch_neurosynth("data/", unpack=True)
ns_dset = nimare.io.convert_neurosynth_to_dataset(
    "data/database.txt",
    "data/features.txt",
)
```
<!-- #endregion -->

```python
ns_dset = nimare.dataset.Dataset.load(op.join(DATA_DIR, "neurosynth_dataset.pkl.gz"))
```

```python
sleuth_dset = nimare.io.convert_sleuth_to_dataset(op.join(DATA_DIR, "sleuth_dataset.txt"))
```

## Searching large datasets

The `Dataset` class contains multiple methods for selecting subsets of studies within the dataset.

One common approach is to search by "labels" or "terms" that apply to studies. In Neurosynth, labels are derived from term frequency within abstracts.

```python
pain_ids = ns_dset.get_studies_by_label("Neurosynth_TFIDF__pain", label_threshold=0.001)
ns_pain_dset = ns_dset.slice(pain_ids)
```

A MACM (meta-analytic coactivation modeling) analysis is generally performed by running a meta-analysis on studies with a peak in a region of interest.

```python
sphere_ids = ns_dset.get_studies_by_coordinate([[24, -2, -20]], r=6)
sphere_dset = ns_dset.slice(sphere_ids)
```

## Running meta-analyses

### Coordinate-based meta-analysis

```python
mkda_kernel = nimare.meta.kernel.MKDAKernel(r=10)
mkda_ma_maps = mkda_kernel.transform(sleuth_dset, return_type="image")
kda_kernel = nimare.meta.kernel.KDAKernel(r=10)
kda_ma_maps = kda_kernel.transform(sleuth_dset, return_type="image")
ale_kernel = nimare.meta.kernel.ALEKernel(sample_size=20)
ale_ma_maps = ale_kernel.transform(sleuth_dset, return_type="image")

# Let's show the kernels
fig, axes = plt.subplots(ncols=3, figsize=(20, 5))
plotting.plot_stat_map(
    mkda_ma_maps[28],
    annotate=False,
    axes=axes[0],
    cmap="Reds",
    cut_coords=[30, -30, -14],
    draw_cross=False,
    figure=fig,
    title="MKDA Kernel",
)
plotting.plot_stat_map(
    kda_ma_maps[28],
    annotate=False,
    axes=axes[1],
    cmap="Reds",
    cut_coords=[30, -30, -14],
    draw_cross=False,
    figure=fig,
    title="KDA Kernel",
)
plotting.plot_stat_map(
    ale_ma_maps[28],
    annotate=False,
    axes=axes[2],
    cmap="Reds",
    cut_coords=[30, -30, -14],
    draw_cross=False,
    figure=fig,
    title="ALE Kernel",
)
fig.show()
```

```python
ale_meta = nimare.meta.cbma.ale.ALE(null_method="approximate")
ale_results = ale_meta.fit(sleuth_dset)
ale_results.maps
```

```python
plotting.plot_stat_map(ale_results.get_map("z"))
```

```python
mc_corrector = nimare.correct.FWECorrector(
    method="montecarlo", 
    n_iters=100, 
    n_cores=1,
)
mc_results = mc_corrector.transform(ale_results)
mc_results.maps
```

```python
plotting.plot_stat_map(mc_results.get_map("z_level-cluster_corr-FWE_method-montecarlo"))
```

### Image-based meta-analysis

```python
pain_dset.images.head()
```

```python
# Calculate missing images
pain_dset.images = nimare.transforms.transform_images(
    pain_dset.images,
    target="z",
    masker=pain_dset.masker,
    metadata_df=pain_dset.metadata,
)
pain_dset.images = nimare.transforms.transform_images(
    pain_dset.images,
    target="varcope",
    masker=pain_dset.masker,
    metadata_df=pain_dset.metadata,
)
```

```python
dsl_meta = nimare.meta.ibma.DerSimonianLaird()
dsl_results = dsl_meta.fit(pain_dset)
```

```python
plotting.plot_stat_map(dsl_results.get_map("z"))
```

```python
ols_meta = nimare.meta.ibma.PermutedOLS()
ols_results = ols_meta.fit(pain_dset)
```

```python
plotting.plot_stat_map(ols_results.get_map("z"))
```

```python

```
