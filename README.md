<h1 align="center">IES: Instance-level Explanation Saliency </h1>

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

This work is part of the project MSxplain, and a pre-print is available at https://doi.org/10.48550/arXiv.2406.09335 (arXiv).

## Author

👤 **Federico Spagnolo**

- Github: [@federicospagnolo](https://github.com/federicospagnolo)
- | [LinkedIn](https://www.linkedin.com/in/federico-spagnolo/) |

## References

1. Spagnolo, F., Molchanova, N., Schaer, R., Bach Cuadra, M., Ocampo Pineda,
M., Melie-Garcia, L., Granziera, C., Andrearczyk, V., Depeursinge, A.: Instance-
level quantitative saliency in multiple sclerosis lesion segmentation. arXiv (2024).
https://doi.org/10.48550/ARXIV.2406.09335
