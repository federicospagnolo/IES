The script ies.py provides the instance-level explanation maps, obtained with methods based on SmoothGrad (average aggregation method) and GradCAM++.
The script ies_maxpool.py provides the same maps, but the first uses the max aggregation method.

To build the environment, an installation of conda or miniconda is needed. Please use "conda env create -f environment.yml", using the provided .yml file.
Usage is described in the first lines of the scripts: "python {FILENAME}.py --model_checkpoint model_epoch_31.pth --input_val_paths {PATH_TO_INPUT1} {PATH_TO_INPUT2} --input_prefixes {INPUT1_FILENAME} {INPUT2_FILENAME} --num_workers 0 --cache_rate 0.01 --threshold 0.3"

