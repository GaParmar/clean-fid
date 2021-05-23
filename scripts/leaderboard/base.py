import os, sys, pdb, pickle, json
import importlib
import argparse
import numpy as np
import torch
import torch.nn as nn
from cleanfid import fid, downloads_helper

def set_random_seed(random_seed):
    torch.cuda.manual_seed(random_seed)
    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)

def make_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--config_file", required=True)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--exp_idx", required=True)
    parser.add_argument("--output_table", required=True)
    parser.add_argument("--metric", default="FID")
    return parser

if __name__=="__main__":
    parser = make_args()
    args = parser.parse_args()
    exp_config = importlib.import_module(args.config_file).L_EXPS[int(args.exp_idx)]

    if "data-efficient-gans" in exp_config["model_url"]:
        if "/stylegan2" in exp_config["model_url"]:
            p = os.path.join(sys.path[0],'tmp',
                    'data-efficient-gans',
                    'DiffAugment-stylegan2-pytorch')
            if p not in sys.path: sys.path.append(p)
            import dnnlib
            import legacy
        elif "/biggan" in exp_config["model_url"]:
            p = os.path.join(sys.path[0],'tmp',
                    'data-efficient-gans',
                    'DiffAugment-biggan-cifar')
            if p not in sys.path: sys.path.append(p)
            import dnnlib
            import utils
        
    else:
        p = os.path.join(sys.path[0],'tmp','stylegan2-ada-pytorch')
        if p not in sys.path: sys.path.append(p)
    
    if not os.path.exists(args.output_table):
        # initialize the logfile as empty
        with open(args.output_table, 'w') as fp:
            json.dump({}, fp, indent=4,sort_keys=True)

    with open(args.output_table, "r") as f: 
        data = json.load(f)


    if f"EXP_{args.exp_idx}" not in data or \
            f"{args.mode}_{args.seed}" not in data[f"EXP_{args.exp_idx}"] or \
            args.metric=="KID" and f"KID_{args.mode}_{args.seed}" not in data[f"EXP_{args.exp_idx}"]:
        # load the GAN model
        if "data-efficient-gans" in exp_config["model_url"]:
            if "/stylegan2" in exp_config["model_url"]:
                with dnnlib.util.open_url(exp_config["model_url"]) as f:
                    G = legacy.load_network_pkl(f)['G_ema'].cuda()
                zdim=512
            elif "/biggan" in exp_config["model_url"]:
                pdb.set_trace()
                parser = utils.prepare_parser()
                config = vars(parser.parse_args(""))
                config['resolution'] = int(exp_config["dataset_res"])
                if exp_config["dataset_name"]=="cifar10":
                    config['n_classes'] = 10
                config['G_activation'] = nn.ReLU(inplace=False)
                config['D_activation'] = nn.ReLU(inplace=False)
                config = utils.update_config_roots(config)
                config['skip_init'] = True
                config['no_optim'] = True
                device = 'cuda'
                model = __import__(config['model'])
                G = model.Generator(**config).cuda()
                G.load_state_dict(torch.load(dnnlib.util.open_file_or_url(exp_config["model_url"])))
                zdim = G.dim_z
        elif "stylegan2-ada-pytorch" in exp_config["model_url"]:
            path = downloads_helper.check_download_url("tmp", exp_config["model_url"])
            with open(path, 'rb') as f:
                G = pickle.load(f)['G_ema'].cuda()
            zdim = 512
        G = torch.nn.DataParallel(G)
        G.eval()
        if "conditional" in exp_config["task_name"]:
              def fg(z):
                bs = z.shape[0]
                c = torch.zeros((bs, 10)).float().cuda()
                c2 = torch.zeros((bs)).int().cuda()
                for _i in range(bs):
                    l = np.random.randint(10)
                    c[_i,l]=1.0
                    c2[_i]=l
                return (G(z,G.module.shared(c2)) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        else:
            fg = lambda z: (G(z,None) * 127.5 + 128).clamp(0, 255).to(torch.uint8)

        set_random_seed(args.seed)
        if args.metric=="FID":
            score = fid.compute_fid(gen=fg, num_workers=0, batch_size=args.batch_size,
                        dataset_name=exp_config["dataset_name"],
                        dataset_res=exp_config["dataset_res"],
                        dataset_split=exp_config["dataset_split"],
                        mode=args.mode,
                        num_gen=exp_config["num_generated_images"],
                        z_dim=zdim)
        elif args.metric=="KID":
            score = fid.compute_kid(gen=fg, num_workers=0, batch_size=args.batch_size,
                        dataset_name=exp_config["dataset_name"],
                        dataset_res=exp_config["dataset_res"],
                        dataset_split=exp_config["dataset_split"],
                        mode=args.mode,
                        num_gen=exp_config["num_generated_images"],
                        z_dim=zdim)
        print(score)
        with open(args.output_table) as fp:
            data = json.load(fp)
        if f"EXP_{args.exp_idx}" not in data: data[f"EXP_{args.exp_idx}"]={}
        for k in exp_config:
            data[f"EXP_{args.exp_idx}"][k] = exp_config[k]
        if args.metric=="FID":
            data[f"EXP_{args.exp_idx}"][f"{args.mode}_{args.seed}"] = score
        elif args.metric=="KID":
            data[f"EXP_{args.exp_idx}"][f"KID_{args.mode}_{args.seed}"] = score
        with open(args.output_table, 'w') as fp:
            json.dump(data, fp, indent=4,sort_keys=True)