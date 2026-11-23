import json
import os
import numpy as np
from pathlib import Path,PurePath
from itertools import batched
from math import ceil
#keep cwd
cwd = os.getcwd()
# cd to \helpers\
helper_path = PurePath(__file__).parent
os.chdir(helper_path)


with open("jobs_config.json" ,"r") as jobs_config_file:
    jobs_config = json.load(jobs_config_file)
""" config example structure
"output_file" : "jobs.json"
"chunks": 4,
"min_size" : 100,
"max_size" : 4000,
"size_increment" : 100,
"min_fill_chance" : 0.5
"max_fill_chance" : 0.95
"fill_chance_increment" : 0.05
"min_survival_chance" : 0.1,
"max_survival_chance" : 1.0,
"min_update_chance" : 0.1
"max_update_chance" : 1.0
"update_chance_increment" : 0.05
"generations" : 500,
"wrap" : true,
"trials_per_sim" : 5
"""

#cd back to script launch path
os.chdir(cwd)




jobs = []
for size in np.arange(jobs_config["min_size"],jobs_config["max_size"]+jobs_config["size_increment"],jobs_config["size_increment"]):
    for fill in np.arange(jobs_config["min_fill_chance"],jobs_config["max_fill_chance"]+jobs_config["fill_chance_increment"],jobs_config["fill_chance_increment"]):
        for update in np.arange(jobs_config["min_update_chance"],jobs_config["max_update_chance"]+jobs_config["update_chance_increment"],jobs_config["update_chance_increment"]):
            for survival in np.arange(jobs_config["min_survival_chance"],jobs_config["max_survival_chance"]+jobs_config["survival_chance_increment"],jobs_config["survival_chance_increment"]):
                job = {
                    "trials": int(jobs_config["trials_per_sim"]),
                    "generations": int(jobs_config["generations"]),
                    "height": int(size),
                    "width": int(size),
                    "wrap": bool(jobs_config["wrap"]),
                    "fill_chance": min(float(fill),1.0),
                    "survival_chance": min(float(survival),1.0),
                    "update_chance": min(float(update),1.0)
                }
                jobs.append(job)


output = jobs_config["output_file"]

chunks: int = jobs_config["chunks"]
chunk_size = ceil(len(jobs)/chunks)
for (i,chunk) in enumerate(batched(jobs, chunk_size)):
    print(output + str(i)+ ".json")
    with open(output + str(i)+ ".json", "w") as file:
        json.dump(chunk, file)
    