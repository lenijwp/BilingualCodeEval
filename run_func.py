import os
from tqdm import tqdm,trange
from utils import stream_jsonl, read_problems, write_jsonl
import warnings
from utils.execution import check_correctness,measure_time
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Union, Iterable, Dict
import itertools
import time
import random
import math
import numpy as np
import pickle
import argparse

def evaluate_program_correctness(
    sample_file: str,
    problem_file: str,
    timeout: float = 10.0,
    n_workers: int = 4,
    repeat: int = 50,
    output_path: str = None
):
    """
    Evaluates the functional correctness of generated samples
    """
    if output_path == None:
        input_name = sample_file.split('/')[-1].split('.')[0]
        output_path = f"./passed/{input_name}_passed.jsonl"

    """
    Evaluates the functional correctness of generated samples, and writes
    results to f"{sample_file}_results.jsonl.gz"
    """

    problems = read_problems(problem_file)

    # Check the generated samples against test suites.
    with ThreadPoolExecutor(max_workers=n_workers) as executor:

        futures = []
        completion_id = Counter()
        n_samples = 0
        results = defaultdict(list)

        print("Reading samples...")
        sample_maxn_adp_dict = {}
        for sample in tqdm(stream_jsonl(sample_file)):
            if sample["task_id"] not in problems:
                continue
            task_id = sample["task_id"]
            completion = sample["completion"]
            # if sample["task_id"] not in sample_maxn_adp_dict:
            #     max_n = int(problems[task_id]['max_n'])
            #     args_ = (problems[task_id], completion, 20, 10, max_n, completion_id)
            #     adpater_ref = measure_time(*args_)
            #     if adpater_ref['time']>timeout/2:
            #         for i in range(1,32):
            #             if adpater_ref['time']/(2**i)<(timeout/2) and adpater_ref['time']/(2**i)>1000:
            #                 max_n = int(max_n/(2**i))
            #                 sample_maxn_adp_dict[sample["task_id"]]=max_n
            #                 break
            #         if task_id not in sample_maxn_adp_dict:
            #             sample_maxn_adp_dict[sample["task_id"]]=int(problems[task_id]['max_n'])
            #     else:
            #         sample_maxn_adp_dict[sample["task_id"]]=int(problems[task_id]['max_n'])

            # args_ = (problems[task_id], completion, timeout, repeat, sample_maxn_adp_dict[sample["task_id"]],completion_id[task_id])
            args_ = (problems[task_id], completion, timeout, repeat, min(int(problems[task_id]['max_n']),2000) ,completion_id[task_id])
            future = executor.submit(check_correctness, *args_)
            futures.append(future)
            completion_id[task_id] += 1
            n_samples += 1


        print("Running test suites...")
        for future in tqdm(as_completed(futures), total=len(futures)):
            result = future.result()
            results[result["task_id"]].append((result["completion_id"], result))
            
    
    for result in results.values():
        result.sort()

    # Finally, save the results in one file:
    def combine_results():
        for sample in stream_jsonl(sample_file):
            task_id = sample["task_id"]
            if task_id not in problems:
                continue
            result = results[task_id].pop(0)
            sample["completion_id"] = result[0]
            sample["real_n"] = result[1]["real_n"]
            sample["result"] = result[1]["result"]
            sample["passed"] = result[1]["passed"]
            yield sample


    print(f"Writing results to {output_path}...")
    write_jsonl(output_path, tqdm(combine_results(), total=n_samples))


    return results

def evaluate_program_timecost(
    sample_file: str,
    problem_file: str,
    timeout: float = 5.0,
    n_workers: int = 4,
    repeat: int = 5,
    seg: int = 100,
    output_path: str = None,
):
    """
    Evaluates the execution time cost of generated samples
    """
    if output_path == None:
        input_name = sample_file.split('/')[-1].split('.')[0]
        output_path = f"./time/{input_name}_timecost.pickle"


    problems = read_problems(problem_file)

    passed_samples = []
    print("Selecting passed samples...")
    for sample in tqdm(stream_jsonl(sample_file)):
        if sample["task_id"] not in problems:
            # warnings.warn(f"Task {sample['task_id']} not found in {problem_file}. Skipping...")
            continue
        if sample["passed"] == False:
            continue
        passed_samples.append(sample)
    print("Evaluating with {} passed samples...".format(len(passed_samples)))

    final_results = []


    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        
        completion_id = Counter()
        n_samples = 0
        # results = defaultdict(list)
        results = []


        start_time  = time.time()
        # for sample in tqdm(passed_samples):
        deal_num = 0
        for sample in passed_samples:

            task_id = sample["task_id"]
            completion = sample["completion"]
            completion_id = sample["completion_id"]

            max_n = int(problems[task_id]['max_n'])
            last = -1
            futures = []

            args_ = (problems[task_id], completion, timeout*(2**3), 10 , max_n, completion_id)
            adpater_ref = measure_time(*args_)
            adpater_false = False
            if adpater_ref['time']>timeout:
                for i in range(1,4):
                    if adpater_ref['time']/(2**i)<timeout and max_n/(2**i)>seg:
                        max_n = int(max_n/(2**i))
                        break
                adpater_false = True
            elif adpater_ref['time']<0:
                    adpater_false = True
            # elif adpater_ref['time']<1e-3:
            #         tmp_max_n = max_n
            #         for i in range(1,11):
            #             tmp_max_n = max_n * (2**i)
            #             if adpater_ref['time']* (2**i) > 1e-3:
            #                 break
            #         max_n = tmp_max_n
            if adpater_false == True:
                continue

            for _i in trange(seg):
                param_n = math.ceil(max_n* (_i + 1) / seg)
                if param_n==last:
                    continue
                args_ = (problems[task_id], completion, timeout, repeat, param_n, completion_id)
                future = executor.submit(measure_time, *args_)
                futures.append(future)

            run_time_list = []
            for future in as_completed(futures):
                result = future.result()
                run_time_list.append((result["n"],result["time"]))
            
            run_time_list.sort(key=lambda x:x[0])
            final_results.append({'task_id':task_id,'completion_id':completion_id,'time':run_time_list})
            deal_num += 1
            print(f'Have dealt {deal_num} / {len(passed_samples)} samples, time cost: {time.time()-start_time} s')
            with open(output_path, 'wb') as file:
                pickle.dump(final_results, file)  


    with open(output_path, 'wb') as file:
        pickle.dump(final_results, file)  

    return final_results







if __name__=='__main__':
    
    parser = argparse.ArgumentParser(description='Input')
    parser.add_argument('--func', type=str, choices=['correctness','time'],help='choose the function to run.')
    parser.add_argument('--sample_file', type=str, help='sample file name under ./data/gencodes.')
    parser.add_argument('--problem_file', type=str, help='problem file name under ./data/testsuite.')
    parser.add_argument('--output_file', type=str, default=None, help='output_name')
    parser.add_argument('--timeout', type=float, default=5.0, help='timeout')
    parser.add_argument('--n_workers', type=int, default=4, help='n_workers')
    parser.add_argument('--repeat', type=int, default=50, help='repeat')
    parser.add_argument('--seg', type=int, default=100, help='seg of max_n')

    args = parser.parse_args()
    # sample_file = './data/gencodes/codegen25-7b-instruct-HumanEval_en.jsonl-num10-t1.0.jsonl'
    # problem_file = './data/testsuite/HumanEval_en.jsonl'
    # evaluate_program_timecost(sample_file,problem_file, output_path='./data/time/codegen25-7b-instruct-HumanEval-chs-10_timecost.jsonl')

    

    if args.func == 'correctness':
        evaluate_program_correctness(args.sample_file, args.problem_file, args.timeout, args.n_workers, args.repeat, args.output_file)
    if args.func == 'time':
        evaluate_program_timecost(args.sample_file, args.problem_file, args.timeout, args.n_workers, args.repeat, args.seg, args.output_file)
