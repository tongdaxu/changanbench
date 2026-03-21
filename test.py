from omegaconf import OmegaConf
from cab.utils import instantiate_from_config
import argparse
from tqdm import tqdm

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="test changan bench")
    # logging params
    parser.add_argument("--config", type=str, default="")
    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    # 基础逻辑
    config = OmegaConf.load(args.config)

    datasets_name = config['datasets']
    codecs_name = config['codecs']
    metrics_name = config['metrics']

    datasets = []

    for dataset_name in datasets_name:
        dataset = instantiate_from_config(config[dataset_name])
        datasets.append((dataset_name, dataset))

    # codec ...
    # TODO


    # metric ...
    # TODO

    # ddp
    for cname, codec in codecs:

        for dname, dataset in datasets:
            
            cache_file_name = os.path.join(cache_dir, cname, dname)

            if os.path.exists(cache_file_name):
                imgs = torch.load(os.path.join(cache_file_name, "imgs.pt"))
                recs = torch.load(os.path.join(cache_file_name, "recs.pt"))
                bpps = torch.load(os.path.join(cache_file_name, "bpps.pt"))

            else:

                # distributed sampler
                # dataset split ...
                data_loader = ...

                imgs = []
                recs = []
                bpps = []
            
                for di, data in tqdm(data_loader):
                    img = data["img"]
                    rec, bpp = codec(img)

                    imgs.append(img)
                    recs.append(rec)
                    bpps.append(bpp)

                imgs = torch.cat(imgs)
                recs = torch.cat(recs)
                bpps = torch.cat(bpps)
                
                torch.save(imgs, os.path.join(cache_file_name, "imgs.pt"))
                torch.save(recs, os.path.join(cache_file_name, "recs.pt"))
                torch.save(bpps, os.path.join(cache_file_name, "bpps.pt"))

            for mname, metric in metrics:
                metric_value = metric(imgs, recs)

    # log format ... -> yaml 