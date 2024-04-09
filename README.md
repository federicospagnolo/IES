<h1 align="center">IES: Instance-level Explanation Saliency 👋</h1>
<p align="center">
  <img src="https://img.shields.io/npm/v/readme-md-generator.svg?orange=blue" />
  <a href="https://www.npmjs.com/package/readme-md-generator">
    <img alt="downloads" src="https://img.shields.io/npm/dm/readme-md-generator.svg?color=blue" target="_blank" />
  </a>
  <a href="https://github.com/kefranabg/readme-md-generator/blob/master/LICENSE">
    <img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-yellow.svg" target="_blank" />
  </a>
  <a href="https://codecov.io/gh/kefranabg/readme-md-generator">
    <img src="https://codecov.io/gh/kefranabg/readme-md-generator/branch/master/graph/badge.svg" />
  </a>
  <a href="https://github.com/frinyvonnick/gitmoji-changelog">
    <img src="https://img.shields.io/badge/changelog-gitmoji-brightgreen.svg" alt="gitmoji-changelog">
  </a>
  <a href="https://twitter.com/FranckAbgrall">
    <img alt="Twitter: FranckAbgrall" src="https://img.shields.io/twitter/follow/FranckAbgrall.svg?style=social" target="_blank" />
  </a>
</p>

> Scripts to generate explainable saliency maps for semantic/instance segmentation tasks.<br /> `IES` can help explaining decisions of machine learning networks performing segmentation tasks on images.

## 🚀 Usage

First, make sure you have python >=3.9 installed.

To build the environment, an installation of conda or miniconda is needed. Once you have it, please use
```sh
conda env create -f environment.yml
```
to build the tested environment using the provided `environment.yml` file. 

The `script ies.py` provides the instance-level explanation maps, obtained with methods based on SmoothGrad (average aggregation method) and GradCAM++. The script `ies_maxpool.py` provides the same maps using the max aggregation method.
Usage is the following:
```sh
python {FILENAME}.py --model_checkpoint model_epoch_31.pth --input_val_paths {PATH_TO_INPUT1} {PATH_TO_INPUT2} --input_prefixes {INPUT1_FILENAME} {INPUT2_FILENAME} --num_workers 0 --cache_rate 0.01 --threshold 0.3
```

## Code Contributors

This work is part of the project MSxplain, and is currently under review.

## Author

👤 **Federico Spagnolo**

- Github: [@federicospagnolo](https://github.com/federicospagnolo)
