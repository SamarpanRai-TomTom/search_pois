"""Project pipelines."""
from typing import Dict

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline

from search_pois import pipeline

ppl = pipeline.create_pipeline()


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """

    return {
        "__default__": ppl["prepare_ppl"] + ppl["split_ppl"] + ppl["query_bing"],
        "reverse_geo": ppl["rev_geo_ppl"],
        "query_bing": ppl["query_bing"],
    }
