---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.11.2
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

<!-- #region -->
# OHBM 2021 NiMARE tutorial

## What is NiMARE?

![NiMARE banner](images/nimare_banner.png)

[NiMARE](https://nimare.readthedocs.io/en/latest/) is a Python library for performing neuroimaging meta-analyses and related analyses, like automated annotation and functional decoding. The goal of NiMARE is to centralize and standardize implementations of common meta-analytic tools, so that researchers can use whatever tool is most appropriate for a given research question.

There are already a number of tools for neuroimaging meta-analysis:

| <h2>Tool</h2> | <h2>Scope</h2> |
| :------------ | :------------- |
| <a href="https://brainmap.org"><img src="images/brainmap_logo.png" alt="BrainMap" width="400"/></a> | BrainMap includes a suite of applications for (1) searching its manually-annotated coordinate-based database, (2) adding studies to the database, and (3) running ALE meta-analyses. While search results can be extracted using its Sleuth app, access to the full database requires a collaborative use agreement. |
| <a href="https://brainmap.org"><img src="images/neurosynth_logo.png" alt="Neurosynth" width="200"/></a> | Neurosynth provides (1) a large, automatically-extracted coordinate-based database, (2) a website for performing large-scale automated meta-analyses, and (3) a Python library for performing meta-analyses and functional decoding, mostly relying on a version of the MKDA algorithm. The Python library has been deprecated in favor of `NiMARE`. |
| <a href="https://www.neurovault.org"><img src="images/neurovault_logo.png" alt="Neurovault" width="200"/></a> | Neurovault is a repository for sharing unthresholded statistical images, which can be used to search for images to use in image-based meta-analyses. Neurovault provides a tool for basic meta-analyses and an integration with Neurosynth's database for online functional decoding. |
| <a href="https://www.sdmproject.com"><img src="images/sdm_logo.png" alt="SDM" width="200"/></a> | The Seed-based _d_ Mapping (SDM) app provides a graphical user interface and SPM toolbox for performing meta-analyses with the SDM algorithm, which supports a mix of coordinates and images. |
| <a href="https://github.com/canlab/Canlab_MKDA_MetaAnalysis"><img src="images/mkda_logo.png" alt="MKDA" width="200"/></a> | The MATLAB-based MKDA toolbox includes functions for performing coordinate-based meta-analyses with the MKDA algorithm. |

The majority of the above tools are (1) closed source, (2) based on graphical user interfaces, and/or (3) written in a programming language that is rarely used by neuroimagers, such as Java. 

In addition to these established tools, there are always interesting new methods that are described in journal articles, but which are never translated to a well-documented and supported implementation.

NiMARE attempts to consolidate the different algorithms that are currently spread out across a range of tools (or which never make the jump from paper to tool), while still ensuring that the original tools and papers can be cited appropriately.

## NiMARE's design philosophy

NiMARE's API is designed to be similar to that of [`scikit-learn`](https://scikit-learn.org/stable/), in that most tools are custom classes. These classes follow the following basic structure:

1. Initialize the class with general parameters
```python
cls = Class(param1, param2)
```

2. For Estimator classes, apply a `fit` method to a `Dataset` object to generate a `MetaResult` object
```python
result = cls.fit(dataset)
```

3. For Transformer classes, apply a `transform` method to an object to return a transformed version of that object

    - An example transformer that accepts a `Dataset`:
```python
dataset = cls.transform(dataset)
```
    - A transformer that accepts a `MetaResult`:
```python
result = cls.transform(result)
```

## Stability and consistency

NiMARE is currently in alpha development, so we appreciate any feedback or bug reports users can provide. Given its status, NiMARE's API may change in the future.

Usage questions can be submitted to [Neurostars with the 'nimare' tag](https://neurostars.org/tag/nimare), while bug reports and feature requests can be submitted to [NiMARE's issue tracker](https://github.com/neurostuff/NiMARE/issues).
<!-- #endregion -->

# Goals for this tutorial

1. Working with NiMARE meta-analytic datasets
1. Searching large datasets
1. Performing coordinate-based meta-analyses
1. Performing image-based meta-analyses
1. Performing functional decoding using Neurosynth

<!-- #region -->
# Before we start, let's download the necessary data **only** if running locally

The code in the following cell checks whether you have the [data](https://osf.io/u9sqa/), and if you don't, it starts downloading it. 

If you're running this notebook locally or using mybinder, then you will need to download the data. You can copy the code below into a new cell, with the Jupyter magic command `%%bash` at the top of the cell.

If you're running it using binder hosted on neurolibre, then you already have access to the data on neurolibre, and you don't need to run this code snippet.

```bash
DIR=$"../data/nimare_tutorial/"
if [ -d "$DIR" ]; then
    echo "$DIR exists."
else 
    mkdir -p $DIR;
    pip install osfclient
    osf -p u9sqa clone  $DIR;
    echo "Created $DIR and downloaded the data";
fi
```
<!-- #endregion -->

```python
# Import the packages we'll need for this tutorial
%matplotlib inline
import json
import os.path as op
from pprint import pprint

import matplotlib.pyplot as plt
from nilearn import plotting, reporting

import nimare
```

```python
DATA_DIR = op.abspath("../data/nimare_tutorial/osfstorage/")
```

# Basics of NiMARE datasets
NiMARE relies on a specification for meta-analytic datasets named [NIMADS](https://github.com/neurostuff/NIMADS). Under NIMADS, meta-analytic datasets are stored as JSON files, with information about peak coordinates, _relative_ links to any unthresholded statistical images, metadata, annotations, and raw text.

**NOTE**: NiMARE users generally do not need to create JSONs manually, so we won't go into that structure in this tutorial. Instead, users will typically have access to datasets stored in more established formats, like [Neurosynth](https://github.com/neurosynth/neurosynth-data) and [Sleuth](http://brainmap.org/sleuth/) files.


We will start by loading a dataset in NIMADS format, because this particular dataset contains both coordinates and images. This dataset is created from [Collection 1425 on NeuroVault](https://identifiers.org/neurovault.collection:1425), which contains [NIDM-Results packs](http://nidm.nidash.org/specs/nidm-results_130.html) for 21 pain studies.

```python
pain_dset = nimare.dataset.Dataset(op.join(DATA_DIR, "nidm_pain_dset.json"))

# In addition to loading the NIMADS-format JSON file,
# we need to download the associated statistical images from NeuroVault,
# for which NiMARE has a useful function.
dset_dir = nimare.extract.download_nidm_pain(data_dir=DATA_DIR)

# We then notify the Dataset about the location of the images,
# so that the *relative paths* in the Dataset can be used to determine *absolute paths*.
pain_dset.update_path(dset_dir)
```

In NiMARE, datasets are stored in a special `Dataset` class. The `Dataset` class stores most relevant information as properties.

The full list of identifiers in the Dataset is located in `Dataset.ids`. Identifiers are composed of two parts- a study ID and a contrast ID. Within the Dataset, those two parts are separated with a `-`.

```python
print(pain_dset.ids)
```

Most other information is stored in `pandas` DataFrames. The five DataFrame-based attributes are `Dataset.metadata`, `Dataset.coordinates`, `Dataset.images`, `Dataset.annotations`, and `Dataset.texts`.

Each DataFrame contains at least three columns: `study_id`, `contrast_id`, and `id`, which is the combined `study_id` and `contrast_id`.

```python
pain_dset.coordinates.head()
```

```python
pain_dset.metadata.head()
```

```python
pain_dset.images.head()
```

```python
pain_dset.annotations.head()
```

```python
pain_dset.texts.head()
```

Other relevant attributes are `Dataset.masker` and `Dataset.space`.

`Dataset.masker` is a [nilearn Masker object](https://nilearn.github.io/manipulating_images/masker_objects.html#), which specifies the manner in which voxel-wise information like peak coordinates and statistical images are mapped into usable arrays. Most meta-analytic tools within NiMARE accept a `masker` argument, so the Dataset's masker can be overridden in most cases.

`Dataset.space` is just a string describing the standard space and resolution in which data within the Dataset are stored.

```python
pain_dset.masker
```

```python
pain_dset.space
```

<!-- #region -->
Datasets can also be saved to, and loaded from, binarized (pickled) files.

We cannot save files on Binder, so here is the code we would use to save the pain Dataset:

```python
pain_dset.save("pain_dataset.pkl.gz")
```
<!-- #endregion -->

Now for a more common situation, where users want to use NiMARE on data from Neurosynth or a Sleuth file.

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
print(f"There are {len(ns_dset.ids)} studies in the Neurosynth database.")
```

```python
sleuth_dset = nimare.io.convert_sleuth_to_dataset(op.join(DATA_DIR, "sleuth_dataset.txt"))
print(f"There are {len(sleuth_dset.ids)} studies in this dataset.")
```

## Searching large datasets

The `Dataset` class contains multiple methods for selecting subsets of studies within the dataset.

One common approach is to search by "labels" or "terms" that apply to studies. In Neurosynth, labels are derived from term frequency within abstracts.

The `slice` method creates a reduced `Dataset` from a list of IDs.

```python
pain_ids = ns_dset.get_studies_by_label("Neurosynth_TFIDF__pain", label_threshold=0.001)
ns_pain_dset = ns_dset.slice(pain_ids)
print(f"There are {len(pain_ids)} studies labeled with 'pain'.")
```

A MACM (meta-analytic coactivation modeling) analysis is generally performed by running a meta-analysis on studies with a peak in a region of interest, so Dataset includes two methods for searching based on the locations of coordinates: `Dataset.get_studies_by_coordinate` and `Dataset.get_studies_by_mask`.

```python
sphere_ids = ns_dset.get_studies_by_coordinate([[24, -2, -20]], r=6)
sphere_dset = ns_dset.slice(sphere_ids)
print(f"There are {len(sphere_ids)} studies with at least one peak within 6mm of [24, -2, -20].")
```

# Running meta-analyses

## Coordinate-based meta-analysis

Most coordinate-based meta-analysis algorithms are kernel-based, in that they convolve peaks reported in papers with a "kernel". Kernels are generally either binary spheres, as in multi-level kernel density analysis (MKDA), or 3D Gaussian distributions, as in activation likelihood estimation (ALE).

NiMARE includes classes for different kernel transformers, which accept Datasets and generate the images resulting from convolving each study's peaks with the associated kernel.

```python
# Create a figure
fig, axes = plt.subplots(ncols=3, figsize=(20, 5))

# Apply different kernel transformers to the same Dataset
kernels = [
    nimare.meta.kernel.MKDAKernel(r=10),
    nimare.meta.kernel.KDAKernel(r=10),
    nimare.meta.kernel.ALEKernel(sample_size=20),
]

for i_kernel, kernel in enumerate(kernels):
    ma_maps = kernel.transform(pain_dset, return_type="image")

    # Plot the kernel
    plotting.plot_stat_map(
        ma_maps[0],
        annotate=False,
        axes=axes[i_kernel],
        cmap="Reds",
        cut_coords=[0, 0, -24],
        draw_cross=False,
        figure=fig,
        title=type(kernel),
    )

# Show the overall figure
fig.show()
```

Meta-analytic Estimators are initialized with parameters which determine how the Estimator will be run. For example, ALE accepts a kernel transformer (which defaults to the standard `ALEKernel`), a null method, the number of iterations used to define the null distribution, and the number of cores to be used during fitting.

The Estimators also have a `fit` method, which accepts a `Dataset` object and returns a `MetaResult` object. [`MetaResult`s](https://nimare.readthedocs.io/en/latest/generated/nimare.results.MetaResult.html#nimare.results.MetaResult) link statistical image names to numpy arrays, and can be used to produce nibabel images from those arrays, as well as save the images to files.

```python
meta = nimare.meta.cbma.ale.ALE(null_method="approximate")
meta_results = meta.fit(pain_dset)
```

```python
print(type(meta_results))
```

```python
print(type(meta_results.maps))
print("Available maps:")
print("\t- " + "\n\t- ".join(meta_results.maps.keys()))
```

```python
z_img = meta_results.get_map("z")
print(type(z_img))
```

```python
plotting.plot_stat_map(
    z_img,
    draw_cross=False,
    cut_coords=[0, 0, 0],
)
```

## Multiple comparisons correction

Most of the time, you will want to follow up your meta-analysis with some form of multiple comparisons correction. For this, NiMARE provides Corrector classes in the `correct` module. Specifically, there are two Correctors: [`FWECorrector`](https://nimare.readthedocs.io/en/latest/generated/nimare.correct.FWECorrector.html) and [`FDRCorrector`](https://nimare.readthedocs.io/en/latest/generated/nimare.correct.FDRCorrector.html). In both cases, the Corrector supports a range of naive correction options relying on [`statsmodels`' methods](https://www.statsmodels.org/dev/generated/statsmodels.stats.multitest.multipletests.html).

In addition to generic multiple comparisons correction, the Correctors also reference algorithm-specific correction methods, such as the `montecarlo` method supported by most coordinate-based meta-analysis algorithms.

Correctors are initialized with parameters, and they have a `transform` method that accepts a `MetaResult` object and returns an updated one with the corrected maps.

```python
mc_corrector = nimare.correct.FWECorrector(
    method="montecarlo", 
    n_iters=100,
    n_cores=1,
)
mc_results = mc_corrector.transform(meta_results)

# Let's store the CBMA result for later
cbma_z_img = mc_results.get_map("z_level-cluster_corr-FWE_method-montecarlo")
```

```python
print(type(mc_results.maps))
print("Available maps:")
print("\t- " + "\n\t- ".join(mc_results.maps.keys()))
```

```python
plotting.plot_stat_map(
    mc_results.get_map("z_level-cluster_corr-FWE_method-montecarlo"),
    draw_cross=False,
    cut_coords=[0, 0, 0],
    vmax=3,
)
```

```python
# Report a standard cluster table for the meta-analytic map using a threshold of p<0.05
reporting.get_clusters_table(cbma_z_img, stat_threshold=1.65)
```

## Image-based meta-analysis

```python
pain_dset.images
```

Note that "z" images are missing for some, but not all, of the studies.

NiMARE's `transforms` module contains a class, `ImageTransformer`, which can generate images from other images- as long as the right images and metadata are available. In this case, it can generate z-statistic images from t-statistic maps, combined with sample size information in the metadata. It can also generate "varcope" (contrast variance) images from the contrast standard error images.

```python
# Calculate missing images
z_transformer = nimare.transforms.ImageTransformer(target="z", overwrite=False)
pain_dset = z_transformer.transform(pain_dset)

varcope_transformer = nimare.transforms.ImageTransformer(target="varcope", overwrite=False)
pain_dset = varcope_transformer.transform(pain_dset)
```

```python
pain_dset.images.head()
```

Now that we have all of the image types we will need for our meta-analyses, we can run a couple of image-based meta-analysis types.

The `DerSimonianLaird` method uses "beta" and "varcope" images, and estimates between-study variance (a.k.a. $\tau^2$).

```python
meta = nimare.meta.ibma.DerSimonianLaird()
meta_results = meta.fit(pain_dset)
```

```python
plotting.plot_stat_map(
    meta_results.get_map("z"),
    draw_cross=False,
    cut_coords=[0, 0, 0],
)
```

The `PermutedOLS` method uses z-statistic images, and relies on [nilearn's `permuted_ols`](https://nilearn.github.io/modules/generated/nilearn.mass_univariate.permuted_ols.html) tool.

```python
meta = nimare.meta.ibma.PermutedOLS()
meta_results = meta.fit(pain_dset)
```

```python
plotting.plot_stat_map(
    meta_results.get_map("z"),
    draw_cross=False,
    cut_coords=[0, 0, 0],
)
```

```python
mc_corrector = nimare.correct.FWECorrector(method="montecarlo", n_iters=100)
mc_results = mc_corrector.transform(meta_results)
```

```python
print(type(mc_results.maps))
print("Available maps:")
print("\t- " + "\n\t- ".join(mc_results.maps.keys()))
```

```python
plotting.plot_stat_map(
    mc_results.get_map("z_level-voxel_corr-FWE_method-montecarlo"),
    draw_cross=False,
    cut_coords=[0, 0, 0],
    vmax=3,
)
```

```python
# Report a standard cluster table for the meta-analytic map using a threshold of p<0.05
reporting.get_clusters_table(
    mc_results.get_map("z_level-voxel_corr-FWE_method-montecarlo"),
    stat_threshold=1.65,
    cluster_threshold=10,
)
```

## Compare to results from the SPM IBMA extension

![IBMA comparison](images/ibma_comparison.png)

Adapted from [Maumet & Nichols (2014)](https://www.frontiersin.org/10.3389/conf.fninf.2014.18.00025/event_abstract).

```python
plotting.plot_stat_map(
    mc_results.get_map("z_level-voxel_corr-FWE_method-montecarlo"),
    threshold=1.65,
    vmax=3,
    draw_cross=False,
    cut_coords=[0, 0, 0],
)
```

```python
plotting.plot_stat_map(
    cbma_z_img,
    threshold=1.65,
    vmax=3,
    draw_cross=False,
    cut_coords=[0, 0, 0],
)
```

# Meta-Analytic Functional Decoding

Functional decoding refers to approaches which attempt to infer mental processes, tasks, etc. from imaging data. There are many approaches to functional decoding, but one set of approaches uses meta-analytic databases like Neurosynth or BrainMap, which we call "meta-analytic functional decoding." For more information on functional decoding in general, read [Poldrack (2011)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3240863/).

In NiMARE, we group decoding methods into three general types: discrete decoding, continuous decoding, and encoding.

- **Discrete decoding methods** use a meta-analytic database and annotations of studies in that database to describe something discrete (like a region of interest) in terms of those annotations.

- **Continuous decoding methods** use the same type of database to describe an unthresholded brain map in terms of the database's annotations. One example of this kind of method is the Neurosynth-based decoding available on Neurovault. In that method, the map you want to decode is correlated with Neurosynth term-specific meta-analysis maps. You end up with one correlation coefficient for each term in Neurosynth. Users generally report the top ten or so terms.

- **Encoding methods** do the opposite- they take in annotations or raw text and produce a synthesized brain map. One example of a meta-analytic encoding tool is [NeuroQuery](https://neuroquery.org/).


Most of the continuous decoding methods available in NiMARE are too computationally intensive and time-consuming for Binder, so we will focus on discrete decoding methods.
The two most useful discrete decoders in NiMARE are the [`BrainMapDecoder`](https://nimare.readthedocs.io/en/latest/generated/nimare.decode.discrete.BrainMapDecoder.html#nimare.decode.discrete.BrainMapDecoder) and the [`NeurosynthDecoder`](https://nimare.readthedocs.io/en/latest/generated/nimare.decode.discrete.NeurosynthDecoder.html#nimare.decode.discrete.NeurosynthDecoder). Detailed descriptions of the two approaches are available in [NiMARE's documentation](https://nimare.readthedocs.io/en/latest/methods/decoding.html#discrete-decoding), but here's the basic idea:

0. A NiMARE `Dataset` must contain both annotations/labels and coordinates.
1. A subset of studies in the `Dataset` must be selected according to some criterion, such as having at least one peak in a region of interest or having a specific label.
2. The algorithm then compares the frequency of each label within the selected subset of studies against the frequency of other labels in that subset to calculate "forward-inference" posterior probability, p-value, and z-statistic.
3. The algorithm also compares the frequency of each label within the subset of studies against the the frequency of that label in the *unselected* studies from the `Dataset` to calculate "reverse-inference" posterior probability, p-value, and z-statistic.

```python
# Given the sheer size of Neurosynth, we will only use the first 500 studies in this example
ns_dset = ns_dset.slice(ns_dset.ids[:500])

label_ids = ns_dset.get_studies_by_label("Neurosynth_TFIDF__amygdala", label_threshold=0.001)
print(f"There are {len(label_ids)} studies in the Dataset with the 'Neurosynth_TFIDF__amygdala' label.")
```

```python
decoder = nimare.decode.discrete.BrainMapDecoder(correction=None)
decoder.fit(ns_dset)
decoded_df = decoder.transform(ids=label_ids)
decoded_df.sort_values(by="probReverse", ascending=False).head(10)
```

```python
decoder = nimare.decode.discrete.NeurosynthDecoder(correction=None)
decoder.fit(ns_dset)
decoded_df = decoder.transform(ids=label_ids)
decoded_df.sort_values(by="probReverse", ascending=False).head(10)
```

# Exercise: Run a MACM and Decode an ROI

Remember that a MACM is a meta-analysis performed on studies which report at least one peak within a region of interest. This type of analysis is generally interpreted as a meta-analytic version of functional connectivity analysis.

We will use an amygdala mask as our ROI, which we will use to (1) run a MACM using the (reduced) Neurosynth dataset and (2) decode the ROI using labels from Neurosynth.


First, we have to prepare some things for the exercise. You just need to run these cells without editing anything.

```python
ROI_FILE = op.join(DATA_DIR, "amygdala_roi.nii.gz")

plotting.plot_roi(
    ROI_FILE,
    title="Right Amygdala",
    draw_cross=False,
)
```

Below, try to write code in each cell based on its comment.

```python
# First, use the Dataset class's get_studies_by_mask method
# to identify studies with at least one coordinate in the ROI.
```

```python
# Now, create a reduced version of the Dataset including only
# studies identified above.
```

```python
# Next, run a meta-analysis on the reduced ROI dataset.
# This is a MACM.
# Use the nimare.meta.cbma.MKDADensity meta-analytic estimator.
# Do not perform multiple comparisons correction.
```

```python
# Initialize, fit, and transform a Neurosynth Decoder.
```

## After the exercise

Your MACM results should look something like this:

![MACM Results](images/macm_result.png)

And your decoding results should look something like this, after sorting by probReverse:

| Term                            |     pForward |   zForward |   probForward |    pReverse |   zReverse |   probReverse |
|:--------------------------------|-------------:|-----------:|--------------:|------------:|-----------:|--------------:|
| Neurosynth_TFIDF__amygdala      | 4.14379e-113 |  22.602    |      0.2455   | 1.17242e-30 |   11.5102  |      0.964733 |
| Neurosynth_TFIDF__reinforcement | 7.71236e-05  |   3.95317  |      0.522177 | 7.35753e-15 |    7.77818 |      0.957529 |
| Neurosynth_TFIDF__olfactory     | 0.0147123    |   2.43938  |      0.523139 | 5.84089e-11 |    6.54775 |      0.955769 |
| Neurosynth_TFIDF__fear          | 1.52214e-11  |   6.74577  |      0.448855 | 6.41482e-19 |    8.88461 |      0.95481  |
| Neurosynth_TFIDF__age sex       | 0.503406     |   0.669141 |      0.524096 | 3.8618e-07  |    5.07565 |      0.954023 |
| Neurosynth_TFIDF__appraisal     | 0.503406     |   0.669141 |      0.524096 | 3.8618e-07  |    5.07565 |      0.954023 |
| Neurosynth_TFIDF__apart         | 0.503406     |   0.669141 |      0.524096 | 3.8618e-07  |    5.07565 |      0.954023 |
| Neurosynth_TFIDF__naturalistic  | 0.555471     |   0.589582 |      0.52505  | 0.00122738  |    3.23244 |      0.95229  |
| Neurosynth_TFIDF__controls hc   | 0.555471     |   0.589582 |      0.52505  | 0.00122738  |    3.23244 |      0.95229  |
| Neurosynth_TFIDF__morphology    | 0.555471     |   0.589582 |      0.52505  | 0.00122738  |    3.23244 |      0.95229  |
