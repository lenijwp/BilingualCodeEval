import json
import gzip

def read_problems(evalset_file):
    return {task["task_id"]: task for task in stream_jsonl(evalset_file)}

def write_json(path,data):
    with open(path,'w',encoding='utf-8') as f:
        json.dump(data,f,indent=4)

def load_json(path):
    res=[]
    with open(path,mode='r',encoding='utf-8') as f:
        dicts = json.load(f)
        res=dicts
    return res

def load_jsonl(path):
    res=[]
    with open(path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    for line in lines:
        json_obj = json.loads(line)
        res.append(json_obj)
    return res


def write_jsonl(path,data):
    with open(path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write((json.dumps(item, ensure_ascii=False) + "\n"))
        f.write('\n')

def stream_jsonl(filename: str):
    """
    Parses each jsonl line and yields it as a dictionary
    """
    if filename.endswith(".gz"):
        with open(filename, "rb") as gzfp:
            with gzip.open(gzfp, 'rt') as fp:
                for line in fp:
                    if any(not x.isspace() for x in line):
                        yield json.loads(line)
    else:
        with open(filename, "r") as fp:
            for line in fp:
                if any(not x.isspace() for x in line):
                    yield json.loads(line)

if __name__=='__main__':
    # a = load_jsonl('./data/HumanEval-chs-100.jsonl')
    # print(a[0])
    from human_eval.data import write_jsonl, read_problems
    problems = read_problems()
    write_jsonl('./data/HumanEval-ens-100.jsonl',problems)






