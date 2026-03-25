import json
from detectron2.data import DatasetCatalog, MetadataCatalog

def register_sa1b(name, json_file, metadata={}):
    """
    Register a dataset in SA-1B JSON format.

    Args:
        name (str): the name that identifies a dataset, e.g. "sa1b_train"
        json_file (str): path to the json file.
        metadata (dict): extra metadata associated with this dataset.
    """
    DatasetCatalog.register(name, lambda: json.load(open(json_file)))
    MetadataCatalog.get(name).set(
        json_file=json_file, evaluator_type="sam", **metadata
    )

# You might need to adjust the path to your JSON file
JSON_PATH = "datasets/sa_1b_train.json"

register_sa1b("sa_1b_train", JSON_PATH) 