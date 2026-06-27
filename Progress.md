## Codec 
| Codec | YAML | ENV | Owner | Mode | Status | 
|-------|------|--------| --------|--------|--------|
| JPEG | ./config/image_codecs/jpeg_*.yaml | ? | Li siqi | Image | Done | 
| HM | ./config/image_codecs/hm_*.yaml | ? | Li siqi | Image | Done | 
| VTM | ./config/image_codecs/vtm_*.yaml | ? |Li siqi | Image | Done | 
| ELIC | ./config/image_codecs/elic_*.yaml | ? | Li siqi | Image | Done | 
| TCM | ./config/image_codecs/tcm_*.yaml | ? |Li siqi | Image | Done | 
| MLIC++ | ./config/image_codecs/MLIC++_*.yaml | ? |Li siqi | Image | Done | 
| HIFIC | ./config/image_codecs/hific_*.yaml | ? |Li siqi | Image | Done | 
| MS-ILLM | ./config/image_codecs/msillm_*.yaml | ? |Li siqi | Image | Done | 
| DiffEIC | ./config/image_codecs/diffeic_*.yaml | ? |Li siqi | Image | Done | 
| StableCodec | ./config/image_codecs/stablecodec_*.yaml | ? |Li siqi | Image | WIP | 
| PerCo | ./config/image_codecs/perco_*.yaml | ? | Li siqi | Image | WIP | 
| FSQ | ./config/image_codecs/fsq_*.yaml | ? | Li siqi | Image | Done | 
| BSQ | ./config/image_codecs/bsq_*.yaml | ? | Li siqi | Image | Done | 
| VAR | ./config/image_codecs/var_*.yaml | ? | Li siqi | Image | Done | 
| TA-ToK | ./config/image_codecs/tatok_*.yaml | ? | Li siqi | Image | Done | 
| Infinity | ./config/image_codecs/infinity_*.yaml | ? | Li siqi | Image | Done | 
| Cosmos | ./config/image_codecs/cosmos_*.yaml | ? | Li siqi | Image | Done | 
| IBQ | ./config/image_codecs/ibq_*.yaml | ? | Li siqi | Image | Done | 
| FlowMo | ./config/image_codecs/flowmo_*.yaml | ? | Li siqi | Image | Done | 
| SSDD | ./config/image_codecs/ssdd_*.yaml | ? | Li siqi | Image | Done | 
| H264 (?) | ? | ? | Kong han | Video | Done |
| H265 (HM) | ? | ? | Kong han | Video | Done |
| H266 (VVC) | ? | ? | Kong han | Video | Done |
| DCVC | ./config/video_codecs/dcvc_q*.yaml | ? | Shi Yicheng | Video | WIP |
| DCVC-TCM | ./config/video_codecs/dcvc_tcm_q*.yaml | ? | Shi Yicheng | Video | WIP |
| DCVC-HEM | ./config/video_codecs/dcvc_hem_q*.yaml | ? | Shi Yicheng | Video | WIP |
| DCVC-DC | ./config/video_codecs/dcvc_dc_q*.yaml | ? | Shi Yicheng | Video | WIP |
| DCVC-FM | ./config/video_codecs/dcvc_fm_q*.yaml | ? | Shi Yicheng | Video | WIP |
| DCVC-RT | ./config/video_codecs/dcvc_rt_q*.yaml | ? | Shi Yicheng | Video | WIP |
| DHVC | ./config/video_codecs/dhvc_q*.yaml | ? | Shi Yicheng | Video | WIP |
| DCVC-B | ./config/video_codecs/dcvc_b_q*.yaml | ? | Shi Yicheng | Video | WIP |
| DCVC-SDD | ./config/video_codecs/dcvc_sdd_q*.yaml | ? | Shi Yicheng | Video | WIP |
| BRHVC | ./config/video_codecs/brhvc_q*.yaml | ? | Shi Yicheng | Video | WIP |

## Metric
| Metric | YAML | Owner | Mode | Status | 
|-------|------|--------|--------|--------|
| PSNR,SSIM,MSSSIM,DISTS,LPIPS,FID | ./config/image_metrics.yaml | Li siqi | Image | Done | 
| PSNR,SSIM,MSSSIM,DISTS,LPIPS,FID,VGGT | ./config/video_metrics.yaml | Kong Han, Shi Yicheng | Video | Done | 
| FLOPS,Latency,param count | ./config/image_metrics.yaml | Li siqi | Image | WIP | 
| FLOPS,Latency,param count | ? | Kong Han | Video | TODO | 
| 座舱VLM | ? | ? | ? | TODO |

## Dataset 
| Dataset | YAML | Owner | Mode | Status | 
|-------|------|--------| --------|--------|
| Kodak | config/image_datasets/kodak_dataset.yaml | Li Siqi | Image | Done | 
| CLIC 2020 | config/image_datasets/clic_dataset.yaml | Li Siqi | Image | Done | 
| ImageNet Val | config/image_datasets/imagenet_dataset.yaml | Li Siqi | Image | Done | 
| XIPH | ? | Kong Han | Video | Done | 
| HEVC CTC | ? | Kong Han | Video | TODO | 
| ScanNet | config/video_datasets/scannet_dataset.yaml | Shi Yicheng | Video | WIP | 
| UCO3D | config/video_datasets/uco3d_dataset.yaml | Shi Yicheng | Video | WIP | 
| 长安座舱 | ? | ? | ? | TODO | 
| 长安道路 | ? | ? | ? | TODO | 

## Infrastructure
| Dataset | Owner | Status | 
|-------|------|--------| 
| Download weights | ? | ? |
| Download dataset | ? | ? |
| Docker | ? | ? |

## Results
* https://1drv.ms/x/c/1fe0643df5e78ab9/IQDdbRV4uGOlTag-fS7YQYexAemepzUeOoAmYtijVN2ZWiw?e=pJzeqe
