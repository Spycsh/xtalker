# XTalker

XTalker (**X**eon Sad**Talker**) is a faster and optimized implementation of [SadTalker](https://github.com/OpenTalker/SadTalker). It utilizes low precision and parallelism to boost the inference speed by up to **10x** compared to the original implementation on one Xeon CPU (without any GPU used). Now it only optimizes the rendering stage, which is one of the two time-consuming bottlenecks. We may consider optimized the other, namely the enhancing stage in the future. This project is experimental and actively under development. Welcome to any advices and ideas.

## How to use

* Install related dependencies of SadTalker following [README](README_SADTALKER.md)

* Install [Intel Extension For PyTorch](https://github.com/intel/intel-extension-for-pytorch)
 
* Generate the parallelized execution script based on your hardware

```
python generate_distributed_infer.py --slot=<parallelism number>
```

Please change the parallelism number with your expected parallelism on your CPU.

* Run the script

```
bash run_distributed_infer_<parallelism number>.sh
```

## Acknowledgements

XTalker borrows heavily from [SadTalker](https://github.com/OpenTalker/SadTalker). We thank the related authors for their great work!

## Disclaimer

This is not to be used for any commercial purposes and is not an official project from Intel. If you have any problems, please contact sihan.chen@intel.com.
