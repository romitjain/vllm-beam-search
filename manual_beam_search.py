import pdb
from collections import namedtuple
from vllm import LLMEngine, EngineArgs, SamplingParams

import json
import numpy as np
from typing import List
from tqdm import tqdm
from datasets import load_dataset, concatenate_datasets

def extract_user_message(elem):
    elem["user_message"] = elem["messages"][0]['content']
    return elem

def build_dataset() -> List[str]:
    ds = load_dataset("soketlabs/bhasha-sft", "indic")
    ds_eng = ds['train'].filter(
        lambda x: x["language"] == "eng", num_proc=16).select(range(16))

    ds_final = concatenate_datasets([ds_eng])
    ds_final = ds_final.map(extract_user_message)

    keep_cols = ["doc_id", "messages", "user_message"]
    ds_final = ds_final.remove_columns(
        [col for col in ds_final.column_names if col not in keep_cols])
    ds_final = list(ds_final["user_message"])

    print(f"Number of samples: {len(ds_final)}")

    return ds_final

beam_width = 2
num_steps = 512
sampling_params = SamplingParams(temperature=0.8, max_tokens=2 , logprobs=2*beam_width, repetition_penalty=2.0, stop_token_ids=[-1])
BeamState = namedtuple("BeamState", ["request_id", "text", "sampling_params", "cumulative_logprob", "root_request_id"])

max_request_id = 0
active_requests = {}  # req_id: cum_prob
current_beams = []

ds = build_dataset()
for d in ds:
    current_beams.append(BeamState(request_id=max_request_id, text=d, sampling_params=sampling_params, cumulative_logprob=0, root_request_id=str(max_request_id)))
    active_requests[str(max_request_id)] = {
        "cumulative_logprob": 0,
        "text": d,
        "parent_request_id": str(max_request_id),
        "root_request_id": str(max_request_id),
        "finished": False
    }
    max_request_id += 1

engine_args = EngineArgs(
    model='meta-llama/Llama-3.2-1B',
    max_num_seqs=256
)
engine = LLMEngine.from_engine_args(engine_args)

for _ in tqdm(range(num_steps), desc="Beam Search"):
    while current_beams:
        beam = current_beams.pop(0)
        engine.add_request(str(beam.request_id), beam.text, beam.sampling_params)

    request_outputs = engine.step()

    expansion = {}

    # This is a single decoding step
    for request_output in request_outputs:
        req_id = request_output.request_id

        if request_output.finished:
            active_requests[req_id]["finished"] = True
            continue

        root_request_id = active_requests[req_id]["root_request_id"]

        # Get logprob of all requests
        top_candidates = [lp for lp in request_output.outputs[0].logprobs[0].values()]
        top_candidates = [(v.decoded_token, v.logprob) for v in top_candidates]

        # Append to prompt and add the previous cum prob for each candidate
        # this is beam_width * beam_width
        for (t, p) in top_candidates:
            if t is None:
                t = ''
            new_text = request_output.prompt + t

            if root_request_id not in expansion:
                expansion[root_request_id] = []

            expansion[root_request_id].append(
                BeamState(request_id=max_request_id, text=new_text, sampling_params=sampling_params,
                 cumulative_logprob=active_requests[req_id]["cumulative_logprob"]+p, root_request_id=root_request_id)
            )
            active_requests[str(max_request_id)] = {
                "cumulative_logprob": active_requests[req_id]["cumulative_logprob"] + p,
                "text": new_text,
                "parent_request_id": req_id,
                "root_request_id": active_requests[req_id]["root_request_id"]
            }
            max_request_id += 1

    # Filter on beam_width, and get the top candidates only
    for root_request_id, beams in expansion.items():
        expansion[root_request_id] = sorted(beams, key=lambda x: x.cumulative_logprob, reverse=True)[:beam_width]
        current_beams.extend(expansion[root_request_id])

    if not (engine.has_unfinished_requests() or current_beams):
        break

# with open("output_custom.jsonl", "w") as f:
#     for r in request_outputs:
#         # get original prompt
#         tmp_request_id = r.request_id
#         root_request_id = active_requests[tmp_request_id]["root_request_id"]
#         probs = []

#         while active_requests[tmp_request_id]["parent_request_id"] != tmp_request_id:
#             probs.append(active_requests[tmp_request_id]["cumulative_logprob"])
#             tmp_request_id = active_requests[tmp_request_id]["parent_request_id"]

#         probs = list(reversed(probs))
#         probs = np.array(probs)
#         prob_values = np.diff(probs, prepend=0.0)
#         prob_values = prob_values.tolist()
#         f.write(json.dumps({
#             "prompt": active_requests[tmp_request_id]["text"],
#             "text": r.prompt + r.outputs[0].text,
#             "probs": prob_values,
#             "root_request_id": root_request_id
#         }))
#         f.write("\n")

with open("output_final.jsonl", "w") as f:
    for k, v in active_requests.items():
        # get original prompt
        tmp_request_id = k
        root_prompt = active_requests[v["root_request_id"]]["text"]
        probs = []

        while active_requests[tmp_request_id]["parent_request_id"] != tmp_request_id:
            probs.append(active_requests[tmp_request_id]["cumulative_logprob"])
            tmp_request_id = active_requests[tmp_request_id]["parent_request_id"]

        probs = list(reversed(probs))
        probs = np.array(probs)
        prob_values = np.diff(probs, prepend=0.0)
        prob_values = prob_values.tolist()
        f.write(json.dumps({
            "prompt": root_prompt,
            "text": v["text"],
            "probs": prob_values,
            "request_id": k,
            "parent_request_id": v["parent_request_id"],
            "root_request_id": v["root_request_id"]
        }))
        f.write("\n")
