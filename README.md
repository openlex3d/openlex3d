# OpenLex3D Toolbox

Scripts and other tools for the OpenLex3D Benchmark

Mantainers: Christina Kassab, Sacha Morin, Martin Büchner, Matías Mattamala


## Setup

```sh
pip install openlex3d
```

### For development

```sh
pip install -e openlex3d
pre-commit install
```

## Running the evaluation script
(This is a work in progress)
```sh
python openlex3d/scripts/evaluate.py -cp <absolute_path_to_config_folder> -cn <config_filename>
```

For example:
```sh
python openlex3d/scripts/evaluate.py -cp /Users/matias/git/openlex3d/openlex3d/config -cn crop_config
```

Alternative, you can use the installed script:
```sh
ol3_evaluate -cp /Users/matias/git/openlex3d/openlex3d/config -cn eval_config
```

## License
TBD