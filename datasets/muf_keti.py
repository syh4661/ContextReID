
import glob
import re

import os.path as osp

from .bases import BaseImageDataset


class MUF_KETI(BaseImageDataset):
    """
    MUF_KETI

    Dataset statistics:
    # identities: 4101
    # images: 32621 (train) + 11659 (query) + 82161 (gallery)
    # cameras: 15
    """
    dataset_dir = '20230629'

    def __init__(self, root='', verbose=True, pid_begin=0, **kwargs):
        super(MUF_KETI, self).__init__()
        self.pid_begin = pid_begin
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'test') ## Todo should be train
        self.test_dir = osp.join(self.dataset_dir, 'test')
        self.list_train_path = osp.join(self.dataset_dir, 'list_gallery.txt') ## Todo should be train
        self.list_val_path = osp.join(self.dataset_dir, 'list_gallery.txt') ## Todo should be val
        self.list_query_path = osp.join(self.dataset_dir, 'list_query.txt')
        self.list_gallery_path = osp.join(self.dataset_dir, 'list_gallery.txt')

        self._check_before_run()
        train = self._process_dir(self.train_dir, self.list_train_path)
        val = self._process_dir(self.train_dir, self.list_val_path)
        train += val
        query = self._process_dir(self.test_dir, self.list_query_path)
        gallery = self._process_dir(self.test_dir, self.list_gallery_path)
        if verbose:
            print("=> MUF_KETI loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(self.gallery)
    def _cam_map(self,cam_id):
        cam_enum=0
        if cam_id in [10,11,12,13,14,15,16,17]:
            cam_enum=cam_id-10

        elif cam_id in [20,21]:
            cam_enum=cam_id-12

        elif cam_id in [40,41,42,43,44,45,46,47]:
            cam_enum = cam_id - 30
        else:
            raise RuntimeError("'{}' is not available cam".format(cam_id))
        return cam_enum

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.test_dir):
            raise RuntimeError("'{}' is not available".format(self.test_dir))

    def _process_dir(self, dir_path, list_path):
        with open(list_path, 'r') as txt:
            lines = txt.readlines()
        dataset = []
        pid_container = set()
        cam_container = set()
        for img_idx, img_info in enumerate(lines):
            img_path, pid = img_info.split(' ')
            pid = int(pid)-1  # no need to relabel # MUF_KETI start 1
            camid = int(img_path.split('_')[1][1:])
            img_path = osp.join(dir_path, img_path)
            dataset.append((img_path, self.pid_begin+pid, self._cam_map(camid), 0))
            pid_container.add(pid)
            cam_container.add(camid)
        print(cam_container, 'cam_container')
        # check if pid starts from 0 and increments with 1
        for idx, pid in enumerate(pid_container):
            assert idx == pid, "See code comment for explanation"
        return dataset

if __name__ == "__main__":
    from datasets.make_dataloader_clipreid import make_dataloader
    from config import cfg
    import argparse
    import sys
    import os
    sys.path.append(os.getcwd())
    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="../configs/person/vit_clipreid_KETI.yml", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)


