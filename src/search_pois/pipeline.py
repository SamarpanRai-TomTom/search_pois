from kedro.pipeline import Pipeline, node, pipeline

from .nodes import filter_rev_geocode, prepare_data, reverse_geocode, split_data


def create_pipeline(**kwargs):

    prepare_ppl = Pipeline(
        [
            node(
                func=prepare_data,
                inputs=["searches", "parameters"],
                outputs=["searches_POI"],
                name="prepare",
            ),
        ]
    )

    split_ppl = Pipeline(
        [
            node(
                func=split_data,
                inputs=["searches_POI", "parameters"],
                outputs={
                    "fail": "searches_fail",
                    "success": "searches_success",
                    "fail_poi": "searches_fail_poi",
                    "success_addr": "searches_success_addr"
                    },
                name="split",
            ),
        ]
    )

    rev_geo_ppl = Pipeline(
        [
            node(
                func=reverse_geocode,
                inputs=["searches_fail_poi", "parameters"],
                outputs=["searches_fail_poi_with_rev_geo"],
                name="reverse_geocode",
            ),
            node(
                func=filter_rev_geocode,
                inputs=[
                    "searches_fail_poi_with_rev_geo",
                    "parameters"
                    ],
                outputs={
                    "all": "searches_fail_poi_result",
                    "osm_better": "searches_osm_better"
                },
                name="filter_rev_geocode",
            ),

        ]
    )

    return {
            'prepare_ppl': prepare_ppl,
            'split_ppl': split_ppl,
            'rev_geo_ppl': rev_geo_ppl
    }
