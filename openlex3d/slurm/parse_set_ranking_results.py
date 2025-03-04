import sys
import pandas as pd
from pathlib import Path
import yaml

pd.options.display.float_format = "{:,.6f}".format

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
        ranking_result = result.get("ranking", {})  # Extract ranking data
        ranking_result.update(
            dict(algorithm=algorithm, dataset=dataset, scene=scene, top_n=top_n)
        )
        data.append(ranking_result)

# Create DataFrame
df = pd.DataFrame(data)

# Set index and sort
df = df.set_index(["top_n", "algorithm", "dataset", "scene"]).sort_index()

df = df[
    [
        "overall",
        "secondary",
        "secondary_inlier_rate",
        "secondary_overshooting",
        "secondary_undershooting",
        "synonym_inlier_rate",
        "synonym_undershooting",
        "synonyms",
    ]
]

# Print full dataframe
print(df.to_string())

# Save to CSV
df.to_csv("results_set_ranking.csv")

# Print average per dataset
print(df.groupby(["top_n", "algorithm", "dataset"]).mean().to_string())
