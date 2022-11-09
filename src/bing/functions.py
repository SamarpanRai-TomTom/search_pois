import os

from azure.cognitiveservices.search.websearch import WebSearchClient
from msrest.authentication import CognitiveServicesCredentials

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

# Replace with your subscription key.
subscription_key = os.getenv("SUBSCRIPTION_KEY", None)
if not subscription_key:
    raise RuntimeError("Please specify SUBSCRIPTION_KEY as environment variable")
# Instantiate the client and replace with your endpoint.
client = WebSearchClient(
    endpoint="https://poi-bing-search.cognitiveservices.azure.com/",
    credentials=CognitiveServicesCredentials(subscription_key),
)


def search(query: str, latitude: str = None, longitude: str = None):
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
        location = f"lat:{latitude};long={longitude}"
    response = client.web.search(query, location)

    parsed_result = [(result.url, result.name) for result in response.web_pages.value]

    # Remove duplicates
    duplicated_removed = list(set(parsed_result))

    # Remove blocked items|
    duplicated_removed = list(
        filter(
            lambda x: not any([item in x[0] for item in BLOCK_LIST]), duplicated_removed
        )
    )

    return duplicated_removed
