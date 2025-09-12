# -*- coding: utf-8 -*-
# import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import  pandas as pd
from cerebras.cloud.sdk import Cerebras
import yaml
from string import Template
import os
# from datasets  import load_dataset




# @click.command()
# @click.argument('input_filepath', type=click.Path(exists=True))
# @click.argument('output_filepath', type=click.Path())
# def main(input_filepath, output_filepath):
#     """ Runs data processing scripts to turn raw data from (../raw) into
#         cleaned data ready to be analyzed (saved in ../processed).
#     """
#     logger = logging.getLogger(__name__)
#     logger.info('making final data set from raw data')


def build_prompt(path, question, response):
    with open(path / "text.yaml") as f:
        raw_text = f.read()
    variables = {
        "Question": question,
        "Response": response,
    }
    text = Template(raw_text).safe_substitute(variables)
    return text

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    load_dotenv(find_dotenv())

    cerebras_endpoint = "https://api.cerebras.ai/v1/chat/completions"
    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]
    data_dir = project_dir / 'data'
    client = Cerebras(
        api_key=os.environ.get("CEREBRAS_API_KEY_HCKTHON"),  # This is the default and can be omitted
    )
    # print(data_dir)
    df =pd.read_csv(data_dir / "train.csv")
    df.drop(['Keywords', 'ID', 'SE_penalized', 'Cluster_labels'], inplace=True, axis=1)
    print(df.head())
    for col in df.columns:
        print(col, df[col][:5])

    for i in range(1):
        prompt = build_prompt(project_dir, df['Question'][i], df['Answer'][i])
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "user", "content": prompt}
            ],
            model="llama-4-scout-17b-16e-instruct",
        )


    print(df.columns)
    print(df.info())
    print(df.describe())
    print(df.shape)



    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables

    # main()
