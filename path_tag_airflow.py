import os

import requests
import argparse
import json
from tqdm import tqdm

def trigger_dag(pre_task_name, dag_id, endpoint, params, user, password, task_name):
    input_data = dict()
    input_data["conf"] = params
    input_data["dag_run_id"] = pre_task_name + str(task_name)
    headers = {"Content-type": "application/json"}
    url = "https://{}/api/v1/dags/{}/dagRuns".format(endpoint, dag_id)
    response = requests.post(url, data=json.dumps(input_data), auth=(user, password), headers=headers)
    # print(response.text)
    res_status_code =  response.status_code
    retry_count = 0
    while res_status_code >= 400 and retry_count < 10:
        response = requests.post(url, data=json.dumps(input_data), auth=(user, password), headers=headers)
        res_status_code = response.status_code
        retry_count += 1
    return response


def trigger_dag_warper(pre_task_name, base_data_root, condition_data_root, json_file, save_root, cfg_file):
    params = dict()
    params["arg"] = f"""
          set -ex
          pip install /mnt/train2/pnd_data/PersonalData/Ness.hu/AirflowComponents/shapely-2.0.1-cp37-cp37m-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
          pip install /mnt/train2/pnd_data/PersonalData/Ness.hu/AirflowComponents/jmespath-1.0.1-py3-none-any.whl
          pip install /mnt/train2/pnd_data/PersonalData/Ness.hu/AirflowComponents/pyntcloud-0.3.1-py2.py3-none-any.whl
          pip install /mnt/train2/pnd_data/PersonalData/Ness.hu/AirflowComponents/urllib3-1.26.19-py2.py3-none-any.whl
          pip install /mnt/train2/pnd_data/PersonalData/Ness.hu/AirflowComponents/botocore-1.33.13-py3-none-any.whl
          pip install /mnt/train2/pnd_data/PersonalData/Ness.hu/AirflowComponents/s3transfer-0.8.2-py3-none-any.whl
          pip install /mnt/train2/pnd_data/PersonalData/Ness.hu/AirflowComponents/boto3-1.33.1-py3-none-any.whl
          cd /mnt/openpai-team/ness.hu/airflow_workflow/path-nn-tagger/
          mpiexec --allow-run-as-root -np 12 python mpi_process.py --base_data_root {base_data_root}  --condition_data_root {condition_data_root} --json_file {json_file} --save_root {save_root} --cfg_file {cfg_file}
    """

    dag_id = "rspnp_nnpath"
    endpoint = "airflow-robocloud.robosense.cn"  # 正式集群
    user = "operator"
    password = "operator"
    trigger_dag(pre_task_name, dag_id, endpoint, params, user, password, os.path.basename(json_file.rstrip('/')))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Path tag production pipeline.')
    parser.add_argument("--base_data_root", type=str, default=None, help="")
    parser.add_argument("--condition_data_root", type=str, default=None, help="")
    parser.add_argument("--json_path", type=str, default=None, help="")
    parser.add_argument("--split_json_path", type=str, default=None, help="")
    parser.add_argument("--split_num", type=int, default=None, help="")
    parser.add_argument("--save_root", type=str, default=None, help="")
    parser.add_argument("--cfg_file", type=str, default=None, help="")
    parser.add_argument("--pre_task_name", type=str, default=None, help="")
    args = parser.parse_args()

    os.makedirs(args.split_json_path, exist_ok=True)
    with open(args.json_path, 'r') as f:
        json_list = json.load(f)

    task_dic = {}
    for item in tqdm(json_list):
        if item.split('/')[0] not in task_dic:
            task_dic[item.split('/')[0]] = [item]
        else:
            task_dic[item.split('/')[0]].append(item)

    for k, v in tqdm(task_dic.items()):
        with open(os.path.join(args.split_json_path, '{}.json'.format(k)), 'w') as f:
            json.dump(v, f)


    for i, json_path in tqdm(list(enumerate(os.listdir(args.split_json_path)))):
        # if 'part' in json_path:
        #     continue
        trigger_dag_warper(args.pre_task_name,
                            args.base_data_root,
                           args.condition_data_root,
                           os.path.join(args.split_json_path, json_path).strip(),
                           args.save_root,
                           args.cfg_file)
        # break