import sys
import pandas as pd
from pathlib import Path
import yaml

pd.options.display.float_format = "{:,.2f}".format

base_path = Path(sys.argv[1])
data = []


# Walk directories in base path
for result_file in base_path.rglob("results.yaml"):
    scene = result_file.parts[-2]
    dataset = result_file.parts[-3]
    top_n = result_file.parts[-4]
    algorithm = result_file.parts[-5]
    with open(result_file, "r") as f:
        result = yaml.load(f, Loader=yaml.FullLoader)
        result = result["iou"] # Only take IOU results
        result.update(
            dict(algorithm=algorithm, dataset=dataset, scene=scene, top_n=top_n)
        )
        data.append(result)


df = pd.DataFrame(data)

# Pivot dataset and scene as rows.
df = df.set_index(["top_n", "algorithm", "dataset", "scene"]).sort_index()
df = df[["synonyms", "depictions", "vis_sim", "clutter", "incorrect", "missing"]]

# Full df
print(df.to_string())

# Save full df to csv
df.to_csv("results.csv")

# Average per dataset
print(df.groupby(["top_n", "algorithm", "dataset"]).mean().to_string())
