import argparse
import yaml
from munch import Munch
from glob import glob

def get_configs():
    parser = argparse.ArgumentParser('srvc')
    parser.add_argument('config', type=str, help='path to config file')
    parser.add_argument('--customer', type=str, help='customer ID')
    parser.add_argument('--month', type=str, help='month in number')
    parser.add_argument('--recnum', type=str, help='number of recommendations')
    args = parser.parse_args()
    cfg_txt = open(args.config, 'r').read()
    cfg = Munch.fromDict(yaml.safe_load(cfg_txt))
    if args.customer:
        cfg['customer_id'] = args.customer
    else:
        cfg['customer_id'] = -1
    if args.month:
        cfg['month'] = int(args.month)
    else:
        cfg['month'] = -1
    if args.recnum:
        cfg['recnum'] = int(args.recnum)
    else:
        cfg['recnum'] = -1
    return cfg