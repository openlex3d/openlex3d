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

### OpenLex3D Ground Truth
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


## Update Paths
All dataset and evaluation paths are configured in `openlex/config/paths.yaml`. Update `paths.yaml` or alternatively create your own `my_paths.yaml` and append `paths=my_paths` to your commands.

## Predictions
Predictions should be stored in a folder as three files:

```bash
embeddings.npy # (n_objects, n_dim) numpy array with object features
index.npy # (n_points,) numpy array of point to object correspondences. embeddings[index[i]] should give the features of the ith point in point_cloud.pcd
point_cloud.pcd # RGB point cloud with n_points
```

For dense methods, `index.npy` will simply be `np.arange(n_points)`.

## Running the evaluation script

### Segmentation IoU
To compute the top1 IoU metric of a method called `bare` on the `office0` scene of the Replica dataset using a GPU, you can use
```sh
python openlex3d/scripts/evaluate.py -cp <absolute to the openlex/config folder> -cn eval_segmentation evaluation.algorithm=bare dataset=replica dataset.scene=office0 evaluation.topn=1 model.device=cuda:0
```
By default, the script will look for predictions at `base_prediction_path/bare/replica/office0` with `base_prediction_path` being defined in `paths.yaml`. You can instead provide your own prediction path by adding `evaluation.predictions_path=<custom path>` to your command.

The dataset options are `replica`, `scannetpp` and `hm3d`. In this example, results will be saved to `output_path/bare/top_1/replica/office0` where `output_path` is again taken from `paths.yaml`.

You can alternatively use the installed script `ol3_evaluate` with the same arguments.

```sh
ol3_evaluate -cp <absolute to the openlex/config folder> -cn eval_segmentation evaluation.algorithm=bare dataset=replica dataset.scene=office0 evaluation.topn=1 model.device=cuda:0
```

You can use the hydra **multirun** function to sequentially process multiple scenes and top n.
```sh
ol3_evaluate -m -cp <absolute to the openlex/config folder> -cn eval_segmentation evaluation.algorithm=bare dataset=replica dataset.scene=office0,office1 evaluation.topn=1,5 model.device=cuda:0
```

### Segmentation Set Ranking
Add `evaluation.set_ranking=true` to the previous commands.

### Queries

To run the queries evaluation metric:
```sh
python openlex3d/scripts/evaluate_queries.py
```

For queries evaluation metric:
```sh
ol3_queries_evaluate
```

## Visualizer
### Segmentation
The `visualize_results.py` will visualize category predictions using open3d. Assuming we ran the command from the Segmentation IoU section, you can use
```sh
python openlex3d/visualization/visualize_results.py output_path/bare/top_1/replica/office0
```
and follow terminal instructions to visualize label predictions for specific point clouds.

### Queries
TODO

## License
TBD
