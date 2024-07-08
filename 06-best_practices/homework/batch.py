#!/usr/bin/env python
# coding: utf-8

import sys
import pickle
import pandas as pd
import argparse
import os
import s3fs


def read_data(filename):
    
    S3_ENDPOINT_URL = os.getenv('S3_ENDPOINT_URL')
    
    if S3_ENDPOINT_URL:
        
        bucket = 'nyc-duration'
        
        options = {
        'client_kwargs': {
            'endpoint_url': S3_ENDPOINT_URL
        }
        }

        return pd.read_parquet(f's3://{bucket}/{filename}.parquet', storage_options=options)

    else:
        return pd.read_parquet(filename)


def save_data(df, key):
    # Set the S3 endpoint URL, AWS access key ID, and AWS secret access key
    os.environ['S3_ENDPOINT_URL'] = 'http://localhost:4566'
    os.environ['AWS_ACCESS_KEY_ID'] = 'abc'
    os.environ['AWS_SECRET_ACCESS_KEY'] = 'xyz'
    os.environ['AWS_DEFAULT_REGION'] = 'eu-west-1'

    # Create an S3 file system object
    fs = s3fs.S3FileSystem(key=os.getenv('AWS_ACCESS_KEY_ID'), region = os.getenv('AWS_DEFAULT_REGION'), secret=os.getenv('AWS_SECRET_ACCESS_KEY'), client_kwargs={'endpoint_url': os.getenv('S3_ENDPOINT_URL')})

    # Set the bucket name
    bucket = 'nyc-duration'

    # Save the DataFrame to the S3 bucket as a Parquet file
    df.to_parquet(f's3://{bucket}/{key}', filesystem=fs)


def prepare_data(df, categorical):
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')

    return df

def get_input_path(year, month):
    default_input_pattern = 'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet'
    input_pattern = os.getenv('INPUT_FILE_PATTERN', default_input_pattern)
    return input_pattern.format(year=year, month=month)


def get_output_path(year, month):
    default_output_pattern = 's3://nyc-duration-prediction-alexey/taxi_type=fhv/year={year:04d}/month={month:02d}/predictions.parquet'
    output_pattern = os.getenv('OUTPUT_FILE_PATTERN', default_output_pattern)
    return output_pattern.format(year=year, month=month)


def main(year, month):
    
    input_file = get_input_path(year, month)
    output_file = get_output_path(year, month)

    categorical = ['PULocationID', 'DOLocationID']
    
    with open('model.bin', 'rb') as f_in:
        dv, lr = pickle.load(f_in)

    df = read_data(input_file)
    
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')

    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = lr.predict(X_val)
    
    print('predicted mean duration:', y_pred.mean())

    df_result = pd.DataFrame()
    df_result['ride_id'] = df['ride_id']
    df_result['predicted_duration'] = y_pred

    df_result.to_parquet(output_file, engine='pyarrow', index=False)

    return df_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prediction of taxi ride duration')
    parser.add_argument('year', type=str, help='Year chosen from yallow taxi datasets')
    parser.add_argument('month', type=int, help='Month chosen from yallow taxi datasets')

    args = parser.parse_args()

    main(args.year, args.month)