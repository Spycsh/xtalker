import argparse
import psutil

parser = argparse.ArgumentParser()
parser.add_argument('--slot', type=int, default=8, help="the number of parallel process")
parser.add_argument('--driven_audio', type=str, default='andy_say_hello.wav', help="the driven audio")
parser.add_argument('--source_image', type=str, default='andy_grove.jpg', help="the source image or video")
parser.add_argument('--result_dir', type=str, default='./result', help="the directory you want to place the results")
parser.add_argument('--log', action='store_true', help="whether to dump logs")
parser.add_argument('--bf16', action='store_true', help="whether to enable bf16")
parser.add_argument('--enhancer', type=str, default=None, help="Face enhancer, [gfpgan, RestoreFormer]")

args = parser.parse_args()

core_num = psutil.cpu_count(logical=False)
unit = core_num / args.slot
with open("run_distributed_infer_{}.sh".format(args.slot), 'w') as f:
    print("run_distributed_infer_{}.sh".format(args.slot))
    for i in range(args.slot):
        socket_idx = 0 if i < args.slot/2 else 1
        start_core = (int)(i * unit)
        end_core = (int)((i+1) * unit -1)
        dump_log = "" if not args.log else f"2>&1|tee {i}.log"
        bf16 = "" if not args.bf16 else "--bf16"
        enhancer = "" if not args.enhancer else f"--enhancer={args.enhancer}"
        f.write(f"numactl -l -C {start_core}-{end_core} python inference.py --driven_audio {args.driven_audio} --source_image {args.source_image} --result_dir {args.result_dir} --cpu --rank={i} --p_num={args.slot} {bf16} {dump_log} {enhancer} &\n")
    f.write("""wait < <(jobs -p)
rm -rf logs
""")

