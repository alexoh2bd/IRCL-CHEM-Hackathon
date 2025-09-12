# -*- coding: utf-8 -*-
# import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import  pandas as pd
from cerebras.cloud.sdk import Cerebras
# import yaml
from jinja2 import Template
import os
import random
import logging
# from datasets  import load_dataset


system_prompt = "You are an expert chemistry data engineer. Given a chemistry question and a candidate positive passage, generate a *single natural-language instruction* that narrows the relevance definition so that ONLY passages which satisfy a specific, testable chemical requirement remain relevant. The instruction must:- add extra qualifications (e.g., ask for explicit monomer feed ratios, solvent system & casting method, reagent grades, temperature/time, measured yields, or safety handling),- *not* include the answer content (do not leak the passage’s factual answer),- be written in natural free-form language,- follow the requested length and style (short/medium/long/very long) and style tag (persona, negation, background, or generic).Output JSON only: {'instruction':'...','length':'short|medium|long|very long','style':'persona|negation|background|generic'} "

# @click.command()
# @click.argument('input_filepath', type=click.Path(exists=True))
# @click.argument('output_filepath', type=click.Path())
# def main(input_filepath, output_filepath):
#     """ Runs data processing scripts to turn raw data from (../raw) into
#         cleaned data ready to be analyzed (saved in ../processed).
#     """
#     logger = logging.getLogger(__name__)
#     logger.info('making final data set from raw data')


def build_prompt(path, question, response, OtherDocs):
    with open(path / "text.yaml") as f:
        raw_text = f.read()
    styles = ["persona", "negation", "background", "generic"]
    lengths = ["short", "medium", "long", "very long"]
    variables = {
        "Question": question,
        "Response": response,
        "OtherDocs": OtherDocs,
        "STYLE" : random.choice(styles),
        "LENGTH": random.choice(lengths),
    }
    # print(raw_text)
    text = Template(raw_text).render(variables)
    # print(text)
    return text

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    load_dotenv(find_dotenv())

    cerebras_endpoint = "https://api.cerebras.ai/v1/chat/completions"
    model = 'qwen-3-32b'
    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]
    data_dir = project_dir / 'data'
    client = Cerebras(
        api_key=os.environ.get("CEREBRAS_API_KEY_HCKTHON"),  # This is the default and can be omitted
    )
    # print(data_dir)
    df =pd.read_csv(data_dir / "train.csv")
    df.drop(['Keywords', 'ID', 'SE_penalized', 'Cluster_labels'], inplace=True, axis=1)
    # print(df.head())


    for i in range(2):
        prompt = build_prompt(project_dir, df['Question'][i], df['Answer'][i], df['chunk'][i])
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "user",  "content": prompt},
                {"role": "system", "content": system_prompt}
            ],
            model=model,
            response_format={ "type": "json_object"}
        )
        print(f"Round {i} QUESTION: {df["Question"][i]} \nANSWER: {df["Answer"][i]}")
        for i in range(len(chat_completion.choices)):
            print("Instruction", chat_completion.choices[0].message.content)





    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables

    # main()
