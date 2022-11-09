import os
import shutil
import re

import pandas as pd
import numpy as np
from ast import literal_eval
from scipy import spatial
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from sentence_transformers import SentenceTransformer

import logging

log = logging.getLogger(__name__)
#log.warning("Issue warning")
#log.info("Send information")


def prepare_data(df, params):
    df['poi_proba'] = df['poi_proba'].str.replace(',','.').astype(float)

    cols = params['cols']

    count = len(df)
    log.info(f'Count on raw data is {count}')


    df['query']=df['query'].str.lower().str.strip()
    df = df[(df['query']!='home') & (df['query'] != 'work')][cols]


    count = len(df)
    log.info(f'Count  removing home and work is {count}')

    #df = df [df['location'].apply(lambda x: len(str(x))>10)]

    transformer = _get_encoder_model()
    cols_to_encode = params['cols_to_encode']

    for col_name in cols_to_encode:
        df = _encode_string(df, col_name, transformer)

    ref_col = cols_to_encode[0]
    for col_name in cols_to_encode[1:]:
        df[f'sim_{ref_col}_{col_name}'] = df.apply(
    lambda row: 1 - spatial.distance.cosine(
        row[f'{ref_col}_encoded'], row[f'{col_name}_encoded']), axis=1)

    for col in cols_to_encode:
        df = df.drop(f'{col}_encoded',axis=1)


    df.loc[:, 'num_comma_all_queries'] = (
        df.loc[:, 'all_queries'].apply(
            lambda qlist: [
                q.strip().count(',') for q in qlist
                ]    
        )
    )

    df.loc[:,'num_all_queries'] = df.loc[:,'all_queries'].apply(lambda q: len(q))
    
    df.loc[:,'num_char_all_queries'] = df.loc[:,'all_queries'].apply(
        lambda qlist: [len(q) for q in qlist]
        )

    df.loc[:, 'num_words_all_queries'] = (
        df.loc[:, 'all_queries'].apply(
            lambda qlist: [
                len(q.strip().replace(',',' ').split(' ')) if  any([c in q for c in [' ',',']]) else 1 for q in qlist
                ]
        )
    )
  
    return [df]


def split_data(df,params):
    poi_proba_threshold = params['poi_proba_threshold']

    df_fail =df[df['success']==False]
    df_fail_poi = df_fail[df_fail["poi_proba"]>poi_proba_threshold]

    df_success = df[df['success']==True]
    
    df_success_addr= df_success[
        df_success["poi_proba"]<(1-poi_proba_threshold)
        ]

    count = len(df_fail)
    log.info(f'Count for FAILING searches is {count}')

    count = len(df_fail_poi)
    log.info(f'Count for FAILING searches potentially POIs is {count}')

    count = len(df_success_addr)
    log.info(f'Count for success searches clearly not POIs is {count}')


    return {
        'fail': df_fail,
        'success': df_success,
        'fail_poi':df_fail_poi,
        'success_addr': df_success_addr
        }

def reverse_geocode(df, params):
    geolocator = Nominatim(user_agent="geoapiExercises")
    rev_geocode = RateLimiter(geolocator.reverse, min_delay_seconds = params['min_delay_seconds'])

    df = df [df['location'].apply(lambda x: len(str(x))>10)]

    df['location_dict'] = df['location'].apply(lambda x:literal_eval(x))
    df['location_coordinates'] = df['location_dict'].apply(lambda d: str(d['lat'])+','+str(d['lon']))
    df['reverse_geocode'] = df['user_latlon'].apply(lambda x:rev_geocode(x))
    
    return [df]

def filter_rev_geocode(df, params):
    col_geo = params['col_rev_geo']
    df[col_geo] = df['reverse_geocode'].apply(
        lambda l: str(l).split(',')[0].strip()
        )
    regex_starts_with_digit = '^\d'
    # Remove reverse geocoding that starts with number
    df = df[~df[col_geo].str.match(regex_starts_with_digit)]

    count = len(df)
    log.info(f'Count for FAIL POIs after removing reverse geocoding starting with number {count}')

    transformer = _get_encoder_model()


    df = _encode_string(df, 'query',transformer)
    df = _encode_string(df, 'reverse_location',transformer)

    df.loc[:, f'sim_query_{col_geo}'] = df.apply(
        lambda row: 1 - spatial.distance.cosine(
        row[f'query_encoded'], row[f'{col_geo}_encoded']), axis=1)

    df.loc[:, 'country_query'] = (
        df['reverse_geocode'].apply(lambda x: x.split(',')[-1])
        .str.strip()
    )

    df = df.drop(['query_encoded',f'{col_geo}_encoded'],axis=1)

    count = len(df)
    log.info(f'Count before filtering by country is {count}')

    df = df[df['country_query']==params['country']]
    count = len(df)
    log.info(f'Count after filtering by country is {count}')

    osm_better = df[
        df['sim_query_reverse_location']>df['sim_query_poiName']
    ]

    count = len(osm_better)
    log.info(f'Count for better reverse than poiName is {count}')

    return {
        'all': df,
        'osm_better': osm_better
    }

def _get_encoder_model():
    return SentenceTransformer('paraphrase-MiniLM-L6-v2')

def _encode_string(df, col_name, transformer):
    df.loc[:,f'{col_name}_encoded'] = df.loc[:,f'{col_name}'].apply(
        lambda sentence: transformer.encode(str(sentence)) if sentence is not None else None
        )
    return df