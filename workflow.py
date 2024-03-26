
        
import os
import logging


logging.basicConfig(filename = f'./data/log.txt', level=logging.DEBUG,format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('result')

gen_code_dir = './data/gencodes'

test_suite_path = './data/testsuite/testsuite.jsonl'

passed_code_dir = './data/passed'

time_code_dir = './data/time'

seg_num = 50

time_limit_correct = 4
time_limit = 6

log_file_path = f'./data/log.txt'

logs = []

with open(log_file_path, 'r') as file:
    for line in file:
        # 处理每一行的内容
        logs.append(line.strip())  # 这里使用strip()去掉每一行结尾的换行符


logger.info(f'-----------------------------------')
logger.info("Start Running")
logger.info(f'-----------------------------------')

for Model in  ['CodeGeeX','CodeGen2.5','CodeLlama','DeepSeek-Coder','StarCoder','GPT-3.5-Turbo']:
    

    gen_code_model_dir = os.path.join(gen_code_dir, Model)
    
    for gen_type in os.listdir(gen_code_model_dir):



        gen_ = gen_type.split('.jsonl')[0]

       
        
        gen_path = os.path.join(gen_code_model_dir, gen_type)

        if os.path.exists(os.path.join(passed_code_dir, Model))==False:
            os.mkdir(os.path.join(passed_code_dir, Model))
        if os.path.exists(os.path.join(time_code_dir, Model))==False:
            os.mkdir(os.path.join(time_code_dir, Model))
        
        pass_path = os.path.join(passed_code_dir, Model, f'{gen_}_passed.jsonl')
        time_path = os.path.join(time_code_dir, Model, f'{gen_}_seg{seg_num}_t{time_limit}.pickle')


        check_cache = False
        for j in range(len(logs)):
            if f'Running Time Measuring for {gen_} in {Model} with time limit {time_limit} and seg num {seg_num}' in logs[j]:
                if j+1<len(logs) and 'Finished Time Measuring with exit code 0' in logs[j+1]:
                    check_cache = True
                    break
                    
        
        if check_cache==True:
            continue
            
        

        logger.info(f'Running Correnctess Checking for {gen_} in {Model} with time limit {time_limit} and seg num {seg_num}')

        exit_code = os.system(f'python run_func.py --func correctness --sample_file {gen_path} --problem_file {test_suite_path} --timeout {time_limit_correct} --n_workers 6 --repeat 100 --output_file {pass_path}')

        logger.info('Finished Correctness Checking with exit code {}'.format(exit_code))

        logger.info(f'Running Time Measuring for {gen_} in {Model} with time limit {time_limit} and seg num {seg_num}')

        exit_code = os.system(f'python run_func.py --func time --sample_file {pass_path} --problem_file {test_suite_path} --timeout {time_limit} --n_workers 6 --repeat 100 --seg {seg_num} --output_file {time_path}')

        logger.info('Finished Time Measuring with exit code {}'.format(exit_code))

        
