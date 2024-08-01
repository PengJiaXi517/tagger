import os
import json
import mpi4py.MPI as MPI
import datetime
import sys
from loguru import logger
import argparse

from main import TagParse


def gen_function(
    cfg_file, json_file, base_data_root, condition_data_root, save_data_root
):
    world = MPI.COMM_WORLD
    world_rank = world.Get_rank()
    world_size = world.Get_size()
    assert world_size >= 2, world_size

    if world_rank == 0:
        process_master(
            cfg_file, json_file, base_data_root, condition_data_root, save_data_root
        )
    else:
        process_worker()


def process_master(
    cfg_file, json_file, base_data_root, condition_data_root, save_data_root
):
    """
    Master process work
    """
    begin = datetime.datetime.now()

    world = MPI.COMM_WORLD
    world_size = world.Get_size()
    workers_status = ["IDLE"] * (world_size)  # IDLE, WORKING

    tasks_send = 0
    tasks_recv = 0

    with open(json_file, "r") as f:
        files = json.load(f)
    file_stack = files
    logger.info("the total num of files : {}".format(len(files)))

    allinall = []
    allerror = []

    while len(file_stack) > 0 or any(i == "WORKING" for i in workers_status):
        for i_worker in range(1, world_size):
            if len(file_stack) > 0 and workers_status[i_worker] == "IDLE":
                file = file_stack.pop()
                world.send(
                    [
                        cfg_file,
                        base_data_root,
                        condition_data_root,
                        save_data_root,
                        file,
                    ],
                    dest=i_worker,
                )
                workers_status[i_worker] = "WORKING"
                tasks_send += 1
        recv_data = world.recv(source=MPI.ANY_SOURCE)
        tasks_recv += 1
        status, recv_line, world_rank = recv_data
        workers_status[world_rank] = "IDLE"

        if not status:
            workers_status[world_rank] = "IDLE"
            allerror.append(recv_line)
        elif status:
            workers_status[world_rank] = "IDLE"
            allinall.append(recv_line)

        if tasks_recv == len(files) or tasks_recv % 1000 == 0:
            logger.info(
                "{} worker:[{:03d}/{}] task:[S:{:06d} R:{:06d} ALL:{:06d}]".format(
                    datetime.datetime.now() - begin,
                    world_rank,
                    world_size - 1,
                    tasks_send,
                    tasks_recv,
                    len(files),
                )
            )
            sys.stdout.flush()

    for i_worker in range(1, world_size):
        world.send(None, dest=i_worker)

    os.makedirs(
        os.path.join(save_data_root, "logs_{}".format(json_file.split("/")[-2])),
        exist_ok=True,
    )
    if len(allerror) != 0:
        with open(
            os.path.join(
                save_data_root,
                "logs_{}".format(json_file.split("/")[-2]),
                "failed_" + json_file.split("/")[-1],
            ),
            "w",
        ) as f:
            json.dump(allerror, f, indent=4)
    with open(
        os.path.join(
            save_data_root,
            "logs_{}".format(json_file.split("/")[-2]),
            "succeed_" + json_file.split("/")[-1],
        ),
        "w",
    ) as f:
        json.dump(allinall, f, indent=4)

    with open(json_file, "r") as f:
        ori_files = json.load(f)
    collect_root = os.path.join(save_data_root, ori_files[0].split('/')[0])
    output = {}
    for sub_dir in os.listdir(collect_root):
        label_root = os.path.join(collect_root, sub_dir, 'labels')
        if os.path.exists(label_root):
            for label_name in os.listdir(label_root):
                with open(os.path.join(label_root, label_name), 'r') as f:
                    output[label_name.split('.')[0]] = json.load(f)
    with open(os.path.join(collect_root, 'tag.json'), 'w') as f:
        json.dump(output, f)

    end = datetime.datetime.now()
    logger.success(f"Time: {end - begin} | Error num: {len(allerror)}")


def process_worker():
    world = MPI.COMM_WORLD
    world_rank = world.Get_rank()

    while True:
        file = world.recv(source=0)
        if file == None:
            break
        cfg_file, base_data_root, condition_data_root, save_root, json_line = file
        status, line = generate(
            cfg_file, base_data_root, condition_data_root, save_root, json_line
        )
        world.send([status, line, world_rank], dest=0)


def generate(cfg_file, base_data_root, condition_data_root, save_root, json_line):

    try:
        tag_parse = TagParse(cfg_file)
        tag_parse.process(base_data_root, condition_data_root, save_root, json_line)
        return True, os.path.join(base_data_root, json_line)
    except:
        return False, os.path.join(base_data_root, json_line)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="arg parser")
    parser.add_argument("--base_data_root", type=str, default=None, help="")
    parser.add_argument("--condition_data_root", type=str, default=None, help="")
    parser.add_argument("--json_file", type=str, default=None, help="")
    parser.add_argument("--save_root", type=str, default=None, help="")
    parser.add_argument("--cfg_file", type=str, default=None, help="")
    args = parser.parse_args()

    base_data_root = args.base_data_root
    condition_data_root = args.condition_data_root
    json_file = args.json_file
    save_root = args.save_root
    cfg_file = args.cfg_file

    # base_data_root = '/mnt/gpfs/PRD/road_percep_0611_add_curb'
    # condition_data_root = "/mnt/train2/RoadPercep/eric.wang/tmp_data/0704_ego_path_parse/"
    # json_file = '/mnt/train2/ness.hu/data_json/data_0611_w_crub_d/train/all.json'
    # save_root = '/mnt/train2/RoadPercep/eric.wang/path_tag/test'
    # cfg_file = '/mnt/train2/RoadPercep/eric.wang/Code/path-nn-tagger/cfg/config.base.py'

    gen_function(
        args.cfg_file,
        args.json_file,
        args.base_data_root,
        args.condition_data_root,
        args.save_root,
    )
