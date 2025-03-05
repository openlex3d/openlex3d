# OpenLex3D Toolbox

Scripts and other tools for the OpenLex3D Benchmark

Mantainers: Christina Kassab, Sacha Morin, Martin Büchner, Kumaraditya Gupta, Matías Mattamala


## Setup

### For GPU
```sh
pip install openlex3d[gpu]
```

### For CPU
```sh
pip install openlex3d[cpu]
```

### For development
Assuming GPU access
```sh
conda create -n openlex3d-env python=3.11
pip install -e .[gpu]
pre-commit install
```

### OpenLex3D ground truth
TODO.

### Datasets
#### Replica
For predictions, methods should use the trajectories from Nice-SLAM. A download script is available [here](https://github.com/cvg/nice-slam/blob/master/scripts/download_replica.sh).

For computing the metrics, you need to download the [original Replica assets](https://github.com/facebookresearch/Replica-Dataset).
#### ScanNet++
In order to download the data, create an account and application [here](https://kaldir.vc.in.tum.de/scannetpp/).

For predictions, methods should use the iPhone RGB-D images and COLMAP poses. 

For computing the metrics, we use a script provided in the [ScanNet++ toolbox](https://github.com/scannetpp/scannetpp) to sample points on the ground truth mesh. Run the following and save the outputs to a folder called prepared_semantics in your main ScanNet++ data directory:
```sh
python -m semantic.prep.prepare_training_data semantic/configs/prepare_training_data.yml
```
#### Habitat Matterport
TODO


## Update paths in configurations
TODO

## Predictions
Predictions should be stored in a folder as three files:

```bash
embeddings.npy # (n_objects, n_dim) numpy array with object features
index.npy # (n_points,) numpy array of point to object correspondences. embeddings[index[i]] should give the features of the ith point in point_cloud.pcd
point_cloud.pcd # RGB point cloud with n_points
```

For dense methods, `index.npy` will simply be `np.arange(n_points)`.

## Running the evaluation script
(This is a work in progress)
```sh
python openlex3d/scripts/evaluate.py -cp <absolute_path_to_config_folder> -cn <config_filename>
```

For example:
```sh
python openlex3d/scripts/evaluate.py -cp /Users/matias/git/openlex3d/openlex3d/config -cn replica
```

To run the queries evaluation metric:
```sh
python openlex3d/scripts/evaluate_queries.py
```

Alternative, you can use the installed script:
```sh
ol3_evaluate -cp /Users/matias/git/openlex3d/openlex3d/config -cn replica
```

For queries evaluation metric:
```sh
ol3_queries_evaluate
```

## License
TBD
