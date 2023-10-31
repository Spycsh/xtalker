# XTalker

XTalker (**X**eon Sad**Talker**) is a faster and optimized implementation of [SadTalker](https://github.com/OpenTalker/SadTalker). It utilizes low precision and parallelism to boost the inference speed by up to **10x** compared to the original implementation on one Sapphire Rapids (SPR) Xeon CPU (without any GPU used). We now have optimized both the rendering and the enhancing stages, which are the main time-consuming stages for the original SadTalker. This project is experimental and actively under development. Welcome to any advices and ideas.

## How to use

### Acceleration by IPEX bf16

* Install PyTorch

```
pip install torch==2.0.0+cpu torchvision==0.15.1+cpu torchaudio==2.0.1 --index-url --index-url https://download.pytorch.org/whl/cpu
```

* Install related dependencies of SadTalker following [README](README_SADTALKER.md)

* Install [Intel Extension For PyTorch](https://github.com/intel/intel-extension-for-pytorch)

```
python3.8 inference.py --driven_audio xxx.wav --source_image xxx.jpg --result_dir ./results --cpu --bf16
```

### Acceleration by IOMP

Normally IOMP is referred to the Intel openmp library. Here I just use this word to refer to my parallelled implementation of SadTalker on Xeon CPU and together with the `libiomp.so` preloaded.

* (Optional) Preload the optional library to get an optimal speedup

```
export LD_PRELOAD=<PATH TO tcmalloc.so>:<PATH TO libiomp5.so>
```

* Generate the parallelized execution script based on your hardware

**Without enhancer**
```
python generate_distributed_infer.py --slot=<parallelism number> --driven_audio xxx.wav --source_image xxx.jpg
```

**With enhancer**

```
python generate_distributed_infer.py --slot=<parallelism number> --driven_audio xxx.wav --source_image xxx.jpg --enhancer gfpgan
```

Please change the parallelism number with your expected parallelism on your CPU.

IOMP can be used together with IPEX bf16 to get a further acceleration. You just need to add `--bf16` when generating the
`run_distributed_infer_xx.sh`, like following

**Without enhancer**
```
python generate_distributed_infer.py --slot=<parallelism number> --driven_audio xxx.wav --source_image xxx.jpg --bf16
```
**With enhancer**
```
python generate_distributed_infer.py --slot=<parallelism number> --driven_audio xxx.wav --source_image xxx.jpg --bf16 --enhancer gfpgan
```

* Run the script

```
bash run_distributed_infer_<parallelism number>.sh
```


### Acceleration by int8 quantization

Go to branch **int8**

```
pip install neural-compressor
```

```
python3.8 inference.py --driven_audio xxx.wav --source_image xxx.jpg --result_dir ./results  --cpu
```

For the first time the model will be quantized and saved to `generator_int8`, and the following runs will load the quantized model from `generator_int8` to do inference.

IOMP can be used together with int8 quantization (powered by [INC](https://github.com/intel/neural-compressor)) to get a further acceleration.

### int8 + IOMP

Go to branch **int8**

```
pip install neural-compressor
```


```
python generate_distributed_infer.py --slot=<parallelism number>
```

The recommended parallelism number to take is to increase from 4 to 8, 16...

Again, remember to run at least twice the following command, when the first run (take a bit longer) is to dump the int8 model, and then in following runs (faster) we can just do inference.

```
bash run_distributed_infer_<parallelism number>.sh
```

int8 + IOMP still have some accuracy loss (maybe because of insufficient calibration or int8 precision itself), which is mainly manifested by noises in the generated frames. This is still under developing.


### Acceleration by using PIRender

Based on Discussion [#457](https://github.com/OpenTalker/SadTalker/discussions/457), PIRender is proved to be faster than facevid2vid for the face rendering stage, and it is orthogonal to our optimization. We integrate this feature into xtalker and apply our optimization to it to get further speedup. To use it, you can simply add a parameter `--facerender pirender` to the above commands.

```
python generate_distributed_infer.py --slot=<parallelism number> --driven_audio=xxx.wav --source_image=xxx.jpg --facerender=pirender
bash run_distributed_infer_<parallelism number>.sh
```

> Notice: PIRender model now has accuracy issue in bf16 optimization so we disable bf16 here. If you want to involve this model, please also follow PIRender's [license](https://github.com/RenYurui/PIRender/blob/main/LICENSE.md) for appropriate usage.

### GC

You can at any time do "garbage collection" when there is an abnormal exit by the following command:

```
bash gc.sh
```

## Benchmarking

We compare the end-to-end inference time (in seconds) between XTalker on SPR and SadTalker with one A100 40GB. The input audio is a 20-second audio file and an image and the output is a 20-second video file. The figures show that our best implementation (IOMP + bf16) is **1.33x** faster than A100 with almost no accuracy loss.

| A100 (Baseline) | SPR fp32 | SPR bf16 | SPR IOMP (--slot=8) + bf16  |
| --- | --- | --- | --- |
| 30.30  | 151.54 | 62.09 | **22.79** |

> Notice: All SPR experiments are ran with the `libiomp.so` preloaded.

## Acknowledgements

XTalker is adapted from [SadTalker](https://github.com/OpenTalker/SadTalker). We thank the related authors for their great work!

## Disclaimer

This is not to be used for any commercial purposes and is not an official project from Intel. If you have any problems, please contact sihan.chen@intel.com.

