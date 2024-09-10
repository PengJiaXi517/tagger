import argparse
import os
import os.path as osp
import random

import boto3
from cyber_record.record import Record
from tqdm import tqdm


class ObjectStorage:
    def __init__(self, bucket_name, s3_root=None, buffer_root=None):
        self.AWS_ACCESS_KEY_ID = "UI2WLBBFHV1CE0HZMZG2"
        self.AWS_SECRET_KEY = "sA2ozSrzC1lehJLbB4AuGCwiuLCHVLOcUbn16xiM"
        node_number = random.randint(81, 96)
        self.ENDPOINT_URL = f"http://10.199.199.{node_number}:8082/"
        self.bucket_name = bucket_name
        self.init_s3_client()
        self.s3_root = s3_root
        self.buffer_root = buffer_root
        if s3_root is not None and buffer_root is not None:
            self.local_root = osp.join(self.buffer_root, osp.basename(s3_root))

    def init_s3_client(self):
        self.s3_client = boto3.client(
            "s3",
            endpoint_url=self.ENDPOINT_URL,
            aws_access_key_id=self.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=self.AWS_SECRET_KEY,
            verify=False,
            use_ssl=False,
        )

    def get_object_names(self, object_key):
        if not object_key.endswith("/"):
            object_key = object_key + "/"
        is_truncated = True
        continuation_token = ""
        filenames = []
        foldernames = []
        while is_truncated:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=object_key,
                Delimiter="/",
                ContinuationToken=continuation_token,
            )
            if "Contents" in response:
                for obj in response["Contents"]:
                    filenames.append(obj["Key"].replace(object_key, ""))
            if "CommonPrefixes" in response:
                for obj in response["CommonPrefixes"]:
                    foldernames.append(obj["Prefix"].replace(object_key, "")[:-1])
            is_truncated = response.get("IsTruncated", False)
            continuation_token = response.get("NextContinuationToken", "")
        return filenames, foldernames

    def download_file(self, object_key, save_root):
        while object_key.startswith("/"):
            object_key = object_key[1:]

        if save_root.split(".")[-1] != object_key.split(".")[-1]:
            os.makedirs(save_root, exist_ok=True)
            save_file = osp.join(save_root, osp.basename(object_key))
        else:
            os.makedirs(osp.dirname(save_root), exist_ok=True)
            save_file = save_root
        self.s3_client.download_file(self.bucket_name, object_key, save_file)

    def upload_file(self, src_file, object_key):
        if not osp.exists(src_file):
            print("WARNING: src file not exist: " + src_file)
            return
        self.s3_client.upload_file(src_file, self.bucket_name, object_key)

    def upload_folder(self, src_root, object_key):
        if not osp.exists(src_root):
            print("WARNING: src file not exist: " + src_root)
            return

        for root, dirs, files in os.walk(src_root):
            for file in files:
                local_file = osp.join(root, file)
                target_key = object_key + "/" + file
                self.s3_client.upload_file(local_file, self.bucket_name, target_key)
