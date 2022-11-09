import logging
from abc import ABC, abstractmethod

from azure.cognitiveservices.search.websearch import WebSearchClient
from elasticsearch import BadRequestError, Elasticsearch
from msrest.authentication import CognitiveServicesCredentials
from transformers import BertModel, BertTokenizer

log = logging.getLogger(__name__)
ELASTIC_INDEX_NAME = "hotels"
MODEL_NAME = "bert-base-multilingual-cased"
BLOCK_LIST = [
    "yelp",
    "tripadvisor",
    "hotels.com",
    "foursquare",
    "facebook",
    "maps.google",
    "maps.bing",
    "booking.com",
    "michelin.com",
    "youtube",
    "instagram",
    "google.com",
]


class SearchProvider(ABC):
    @abstractmethod
    def search(self):
        pass


def singleton(class_):
    instances = {}

    def getinstance(*args, **kwargs):
        if class_ not in instances:
            instances[class_] = class_(*args, **kwargs)
        return instances[class_]

    return getinstance


@singleton
class BingSearch(SearchProvider):
    def __init__(self, subscription_key) -> None:
        if not subscription_key:
            raise RuntimeError(
                "Please specify SUBSCRIPTION_KEY as environment variable"
            )
        # Instantiate the client and replace with your endpoint.
        self.client = WebSearchClient(
            endpoint="https://poi-bing-search.cognitiveservices.azure.com/",
            credentials=CognitiveServicesCredentials(subscription_key),
        )

    def search(self, query: str, latitude: str = None, longitude: str = None):
        """Searches in Bing Cognitive Web search given the query and optionally location

        Args:
            query (str): The query text
            latitude (str, optional): latitude. Defaults to None.
            longitude (str, optional): longitude. Defaults to None.

        Returns:
            (tuple): List of tuple containing (Url, Name) of result
        """
        location = None
        if latitude and longitude:
            location = f"lat={latitude};long={longitude}"
        response = self.client.web.search(query, location)

        if response.web_pages is not None:
            parsed_result = [
                (result.url, result.name) if result is not None else None
                for result in response.web_pages.value
            ]

            # Remove duplicates
            duplicated_removed = list(set(parsed_result))

            # Remove blocked items|
            duplicated_removed = list(
                filter(
                    lambda x: not any([item in x[0] for item in BLOCK_LIST]),
                    duplicated_removed,
                )
            )
        else:
            duplicated_removed = None

        return duplicated_removed


@singleton
class HotelSearch(SearchProvider):
    def __init__(self, elastic_search_endpoint) -> None:
        self.es = Elasticsearch(elastic_search_endpoint)
        log.info(f"Loading model {MODEL_NAME}. Might take a while..")
        self.tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
        self.model = BertModel.from_pretrained(MODEL_NAME)

    def embed(self, text: str):
        text = text[:512]
        encoded_input = self.tokenizer(text, return_tensors="pt")
        output = self.model(**encoded_input)
        return output.pooler_output[0].tolist()

    def get_query_config(self, query: str):
        query_vector = self.embed(query)
        query_config = {
            "knn": {
                "field": "description-vector",
                "query_vector": query_vector,
                "k": 10,
                "num_candidates": 100,
            },
            "fields": ["name"],
        }

        return query_config

    def search(self, query):
        print("query:{}".format(query))
        es_query = self.get_query_config(query)
        resp = self.es.search(index=ELASTIC_INDEX_NAME, body=es_query)

        # Transform to consistent format between ES/Solr
        matches = []
        for hit in resp["hits"]["hits"]:
            hit["_source"]["_score"] = hit["_score"]
            matches.append(hit["_source"])

        return matches, resp["took"], resp["hits"]["total"]["value"]
