# XTalker

XTalker (**X**eon Sad**Talker**) is a faster and optimized implementation of [SadTalker](https://github.com/OpenTalker/SadTalker). It utilizes low precision and parallelism to boost the inference speed by up to **10x** compared to the original implementation on one Xeon CPU (without any GPU used). Now it only optimizes the rendering stage, which is one of the two time-consuming bottlenecks. We will optimize the other, namely the enhancing stage in the future. This project is experimental and actively under development. Welcome to any advices and ideas.

## How to use

### Acceleration by IPEX bf16

* Install PyTorch

```
pip install torch==2.0.0+cpu torchvision==0.15.1+cpu torchaudio==2.0.1 --index-url --index-url https://download.pytorch.org/whl/cpu
```

* Install related dependencies of SadTalker following [README](README_SADTALKER.md)

* Install [Intel Extension For PyTorch](https://github.com/intel/intel-extension-for-pytorch)

```
python3.8 inference.py --driven_audio xxx.wav --source_image xxx.jpg --result_dir ./result  --cpu
```

### Acceleration by IOMP

* Generate the parallelized execution script based on your hardware

```
python generate_distributed_infer.py --slot=<parallelism number>
```

Please change the parallelism number with your expected parallelism on your CPU.

* Run the script

```
bash run_distributed_infer_<parallelism number>.sh
```

IOMP can be used together with IPEX bf16 to get a further acceleration.

### Acceleration by int8 quantization

Go to branch **int8**

```
pip install neural-compressor
```

```
python3.8 inference.py --driven_audio xxx.wav --source_image xxx.jpg --result_dir ./result  --cpu
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

### Limitation

With int8 + IOMP, the generation speed on one Sapphire Rapids Xeon is close to the generation speed on one A100 (~15it/s). However both of them are still not enough for real-time without other tradeoffs, and some accuracy loss may exist because of using low-precision in this project.

### GC

You can at any time do "garbage collection" when there is an abnormal exit by the following command:

```
bash gc.sh
```

## Acknowledgements

XTalker borrows heavily from [SadTalker](https://github.com/OpenTalker/SadTalker). We thank the related authors for their great work!

## Disclaimer

This is not to be used for any commercial purposes and is not an official project from Intel. If you have any problems, please contact sihan.chen@intel.com.

