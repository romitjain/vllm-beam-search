"""
This script is a naive implementation of beam search for text generation.
It uses the vLLM library to perform beam search.
"""

import time
import json
from typing import List
from argparse import ArgumentParser

from vllm import LLM, SamplingParams
from vllm.sampling_params import BeamSearchParams
from datasets import load_dataset, concatenate_datasets


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-1B-Instruct", required=False)
    parser.add_argument("--max_tokens", type=int, default=128, required=False)
    parser.add_argument("--beam_width", type=int, default=4, required=False)
    parser.add_argument("--temperature", type=float, default=0.8, required=False)

    return parser.parse_args()


def extract_user_message(elem):
    elem["user_message"] = elem["messages"][0]['content']
    return elem


def build_dataset() -> List[str]:
    ds = load_dataset("soketlabs/bhasha-sft", "indic")
    ds_eng = ds['train'].filter(lambda x: x["language"] == "eng", num_proc=16).select(range(8))
    ds_non_eng = ds['train'].filter(lambda x: x["language"] != "eng", num_proc=16).select(range(8))

    ds_final = concatenate_datasets([ds_eng, ds_non_eng])
    ds_final = ds_final.map(extract_user_message)

    keep_cols = ["doc_id", "messages", "user_message"]
    ds_final = ds_final.remove_columns([col for col in ds_final.column_names if col not in keep_cols])
    ds_final = list(ds_final["user_message"])

    print(f"Number of samples: {len(ds_final)}")

    return ds_final


def generate_response(llm: LLM, prompts: List[str], sampling_params: SamplingParams, beam_search_params: BeamSearchParams):
    start_time = time.time()
    outputs = llm.beam_search(prompts, beam_search_params)
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")

    with open("output_naive.jsonl", "w") as f:
        for prompt, output in zip(prompts, outputs):
            prompt_dict = {"prompt": prompt, "beam_candidates": {}}
            for idx, beam in enumerate(output.sequences):
                generated_text = beam.text
                prompt_dict["beam_candidates"][idx] = {
                    "text": generated_text,
                    "cum_logprob": beam.cum_logprob,
                }
            f.write(json.dumps(prompt_dict))
            f.write("\n")


def main():
    args = parse_args()
    llm = LLM(model=args.model)
    sampling_params = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature
    )
    beam_search_params = BeamSearchParams(
        beam_width=args.beam_width,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )

    ds = build_dataset()
    generate_response(llm, ds, sampling_params, beam_search_params)

if __name__ == "__main__":
    main()
