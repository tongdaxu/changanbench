## Image & Video Codec Evaluation

ChanganBench is a configurable benchmark toolkit for large-scale image and video codec evaluation. It provides a unified framework for evaluating traditional codecs, learned codecs, generative codecs and tokenizer-based codecs using distributed execution and configurable benchmarking pipelines.

## Image Codec Evaluation

### Supported Image Codecs

<table>
<tr>
<th width="22%">Perceptual Image Codec</th>
<th width="28%"></th>
<th width="20%">Image Tokenizer</th>
<th width="20%"></th>
<!-- <th width="10%">Datasets</th> -->
<!-- <th width="15%">Metrics</th> -->
</tr>

<tr>
<td><b>Traditional</b></td>
<td>JPEG, HM, VTM</td>
<td rowspan="2"><b>GAN-based</b></td>
<td rowspan="2">
FSQ, BSQ, VAR, TA-Tok<br>
Infinity, Cosmos, IBQ
<!-- </td>
<td rowspan="4">Imagenet CLIC2020 Kodak</td>
<td rowspan="4"> -->
<!-- PSNR<br>
SSIM<br>
LPIPS<br>
DISTS<br>
FID<br>
... -->
</td>
</tr>

<tr>
<td><b>Learning-based</b></td>
<td>ELIC, TCM, MLIC++</td>
</tr>

<tr>
<td><b>GAN-based</b></td>
<td>HiFiC, MS-ILLM</td>
<td rowspan="2"><b>Diffusion-based</b></td>
<td rowspan="2">
FlowMo, SSDD
</td>
</tr>

<tr>
<td><b>Diffusion-based</b></td>
<td>PerCo, DiffEIC, StableCodec</td>
</tr>

</table>

### Supported Metrics

| Category | Metrics |
|-----------|----------|
| Pixel Fidelity | PSNR, SSIM, MS-SSIM |
| Perceptual Quality | LPIPS, DISTS |
| Distribution Quality | FID |
| Compression Efficiency | BPP |

### Supported Datasets

| Dataset | Description |
|----------|-------------|
| Imagenet/ CLIC2020/ Kodak | Built-in benchmark dataset |
| Custom Dataset | Any image directory or dataset implementation |

---

New codecs can be integrated by implementing a codec class and registering it through the configuration system.

```python
reconstruction, bpp = codec(image)
```

---


### Evaluation Pipeline

#### Image Evaluation

```text
Dataset
    ↓
Codec Inference
    ↓
Reconstruction + BPP
    ↓
Metric Evaluation
    ↓
Distributed Aggregation
    ↓
Final Benchmark Results
```

---

## Video Evaluation

### Supported Video Codecs

<table>


<tr>
<td><b>Traditional</b></td>
<td>
H.264, H.265, H.266
</td>
<td rowspan="2"><b>Neural Codec</b></td>
<td rowspan="2">
DCVC, DCVC-TCM, DCVC-HEM<br>
DCVC-DC, DCVC-FM, DCVC-RT
</td>
</tr>



</table>

### Supported Video Metrics

| Category | Metrics |
|-----------|----------|
| Pixel Fidelity | PSNR, SSIM, MS-SSIM |
| Perceptual Quality | LPIPS, DISTS |
| Distribution Quality | FID, FVD |
| 3D/Geometry Sensitivity | VGGT camera center error, camera rotation error, depth AbsRel, point L2 |
| Compression Efficiency | BPP  |


### Configuration-Driven Benchmarking

Each benchmark experiment is defined by three configurable components:

- Dataset
- Codec
- Metric

All components are configured through YAML files, enabling arbitrary combinations of datasets, codecs, and evaluation criteria without modifying benchmark code.

The framework supports:

- PyTorch Distributed Data Parallel (DDP)
- Extensible codec registration
- Extensible metric registration
- Extensible dataset registration
- Unified image and video benchmarking workflows
