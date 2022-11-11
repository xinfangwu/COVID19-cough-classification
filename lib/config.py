import os
from yacs.config import CfgNode as CN

config = CN()
config.ckpt_root = 'ckpt'
config.cuda = True
config.cuda_benchmark = False
config.num_workers = 8

config.dataset = CN()
config.dataset.name = 'CoughDataset'
config.dataset.common_kwargs = CN(new_allowed=True)
config.dataset.train_kwargs = CN(new_allowed=True)
config.dataset.valid_kwargs = CN(new_allowed=True)
config.dataset.test_kwargs = CN(new_allowed=True)

config.training = CN()
config.training.epoch = 50
config.training.batch_size = 20
config.training.save_every = 5
config.training.optim = 'Adam'
config.training.lr = 0.0001
config.training.weight_decay = 5e-4

config.model = CN()
config.model.file = 'lib.model.CoughModel'
config.model.modelclass = 'CoughModel'
config.model.loss_func = ''
config.model.kwargs = CN(new_allowed=True)

config.testing = CN()
config.testing.name = ''
config.testing.batch_size = 1
config.testing.ckpt_root = ''


def update_config(cfg, args):
    cfg.defrost()
    cfg.merge_from_file(args.cfg)
    cfg.freeze()

def infer_exp_id(cfg_path, ckpt_root):
    cfg_path = cfg_path.split('config/')[-1]
    if cfg_path.endswith('.yaml'):
        cfg_path = cfg_path[:-len('.yaml')]
    exp_dir, exp_id = os.path.split(cfg_path)
    exp_dir = os.path.join(ckpt_root, exp_dir)
    exp_id = exp_id.strip('/')
    return exp_dir, exp_id

