import pickle
import json
from tqdm import tqdm
from transformers import pipeline
import argparse
import math


def get_binary_response(passage, query, model_size, exp_model, cheap_model):
    prompt = f"Is the following passage related to the query?\npassage: {passage}\nquery: {query}\nAnswer in yes or no"
    if model_size == 'expensive':
        ans = exp_model(prompt)[0]['generated_text']
    else:
        ans = cheap_model(prompt)[0]['generated_text']

    return ans
def get_prp_response(query, passage1, passage2, model_size, exp_model, cheap_model):
    prompt = f"""Given a query "{query}", which of the following two passages is more relevant to the query?
Passage A: {passage1}
Passage B: {passage2}
Output Passage A or Passage B.
"""
    if model_size == 'expensive':
        ans = exp_model(prompt)[0]['generated_text']
    else:
        ans = cheap_model(prompt)[0]['generated_text']
    return ans

def count_top_l(args,limit, query_len, ctxs, id2t):
    tokens = 0
    howmany = 0
    reached = False

    while True:
        if reached:
            break
        for idx in range(len(ctxs[:args.total_passages])-1):
            text1 = str(id2t[ctxs[idx]['id']]) 
            text1_len = len(text1.split())
            text2 = str(id2t[ctxs[idx + 1]['id']]) 
            text2_len = len(text2.split())

            possible_tokens = args.prp_prompt_head_len + args.prp_output_possible_len+ query_len + text1_len + text2_len
            if tokens + possible_tokens < limit:
                tokens += possible_tokens
                howmany += 1
            else:
                reached = True
                break
    
    return howmany

def run_eco(args):

    # load the dataset
    print(f'loading dataset')
    with open(args.input_dataset, "r") as f:
        data = json.load(f)

    print(f'loading wikipedia dict')
    # load the wikipedia dict
    with open(args.input_wikipedia_dict, "rb") as f:
        id2t = pickle.load(f)

    # load the cheap model
    cheap_model = pipeline("text2text-generation", model=args.cheap_modelcard, device=0)

    # load the expensive model
    exp_model = pipeline("text2text-generation", model=args.exp_modelcard, device=0)

    t = args.budget_tokens

    outermostlist = []

    for data_idx, da in enumerate(tqdm(data)):

        # FIRST STAGE
        BINARY_TOKEN_LIMIT =  int(args.budget_split_x * t) * 1

        binary_running_token = 0

        innerjson = {}
        orig_qstn = da['question']
        orig_qstn_len = len(orig_qstn.split())
        ctxs = da['ctxs']

        yes_ctxs = []
        no_ctxs = []
        reranked_list = []
        ending_idx = 0

        yes_cntr = 0
        howmany = 0

        for idx, ct in enumerate(tqdm(ctxs[:args.total_passages], leave = False)):
            text = str(id2t[ct['id']]) #  retrieve the context given id
            text_len = len(text.split())
            token_reached = False

            if binary_running_token + args.binary_prompt_head_len + text_len + orig_qstn_len + args.binary_output_possible_len < BINARY_TOKEN_LIMIT:

                binary_running_token += args.binary_prompt_head_len + text_len + orig_qstn_len

                try:
                    ans = get_binary_response(text, orig_qstn, 'expensive', exp_model, cheap_model).strip().lower() # call LLM
                except Exception as e:
                    ans = ''
                if "yes" in ans:
                    ct['gen_question'] = ans
                    yes_cntr += 1
                    yes_ctxs.append(ct)
                elif "no" in ans:
                    ct['gen_question'] = ans
                    no_ctxs.append(ct)
                else:
                    ans = ""
                if ans:
                    binary_running_token += len(ans.split())
                else:
                    yes_ctxs.append(ct)
                howmany += 1
            else:
                ending_idx = idx
                token_reached = True
                break

        reranked_list.extend(yes_ctxs)
        if token_reached:
            reranked_list.extend(ctxs[ending_idx:args.total_passages])
        reranked_list.extend(no_ctxs)

        #SECOND STAGE
        PRP_TOKEN_LIMIT =  int(args.budget_split_y * t) * 3

        top_l = count_top_l(args, PRP_TOKEN_LIMIT, orig_qstn_len, reranked_list, id2t) # calculate approx number of passages that can be processed
        full_value = math.ceil(top_l / args.total_passages)
        cntr = full_value
        mod_value = top_l % args.total_passages
        while cntr > 0:
            if cntr > 1:
                limit_val = args.total_passages
            else:
                limit_val = mod_value if mod_value else args.total_passages

            for idx in tqdm(range(len(reranked_list[:limit_val])-1, 0, -1), leave=False):
                pass_b = str(id2t[reranked_list[idx]['id']])
                pass_a = str(id2t[reranked_list[idx - 1]['id']])

                pass_b_vote = 0
                pass_a_vote = 0

                try:
                    ans1 = get_prp_response(orig_qstn, pass_b, pass_a, 'cheap', exp_model, cheap_model).strip().lower() # call LLM
                except Exception as e:
                    ans1 = ""
                if ans1:
                    if ans1 == "passage a":
                        pass_b_vote += 1
                    elif ans1 == "passage b":
                        pass_a_vote += 1
                if pass_b_vote > pass_a_vote:
                    # swap
                    reranked_list[idx], reranked_list[idx-1] = reranked_list[idx-1], reranked_list[idx] 
            cntr -= 1
        
        innerjson['question'] = orig_qstn
        innerjson['answers'] = da['answers']
        innerjson['reranked'] = reranked_list
        outermostlist.append(innerjson)
    
    

    with open(args.output_results, "w") as f:
        json.dump(outermostlist, f)



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_wikipedia_dict", type = str, default = "wiki_id2text.pickle", required = False,
                        help = "Path of the wikipedia passages in pickle format.")
    parser.add_argument("--input_dataset", type = str, default = "./downloads/data/retriever-outputs/bm25/nq-test.json", required = True,
                        help = "Path of the dataset.")
    parser.add_argument("--binary_prompt_head_len", type = int, default = 15, required = False,
                        help = "Pointwise(binary) default prompt length.")
    parser.add_argument("--binary_output_possible_len", type = int, default = 1, required = False,
                        help = "Pointwise(binary) default output length.")
    parser.add_argument("--total_passages", type = int, default = 50, required = False,
                        help = "Total passages to consider.")
    parser.add_argument("--prp_prompt_head_len", type = int, default = 25, required = False,
                        help = "Pairwise default prompt length.")
    parser.add_argument("--prp_output_possible_len", type = int, default = 2, required = False,
                        help = "Pairwise default output length.")
    
    parser.add_argument("--cheap_modelcard", type = str, default = "google/flan-t5-large", required = False,
                        help = "Model card of the cheap model.")
    parser.add_argument("--exp_modelcard", type = str, default = "google/flan-t5-xl", required = False,
                        help = "Model card of the expensive model.")

    parser.add_argument("--budget_tokens", type = int, default = 4000, required = False,
                        help = "Budget in tokens.")
    parser.add_argument("--budget_split_x", type = int, default = 0.5, required = False,
                        help = "Budget percentage allocated for first stage of EcoRank.")
    parser.add_argument("--budget_split_y", type = int, default = 0.5, required = False,
                        help = "Budget percentage allocated for second stage of EcoRank.") 
    
    parser.add_argument("--output_results", type = str, default = "ecorank_output.json", required = False,
                        help = "Path of the wikipedia passages in pickle format.")
    args, unknown = parser.parse_known_args()
    return args

def main():
    args = get_args()
    run_eco(args)
    
if __name__ == "__main__":
    main()