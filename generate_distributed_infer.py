import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--slot', type=int, default=8, help="the number of parallel process")
parser.add_argument('--core', type=int, default=112, help="the number of cpu")
parser.add_argument('--driven_audio', type=str, default='andy_say_hello.wav', help="the driven audio")
parser.add_argument('--source_image', type=str, default='andy_grove.jpg', help="the source image or video")
parser.add_argument('--result_dir', type=str, default='./result', help="the directory you want to place the results")

args = parser.parse_args()

# numactl -m 0 -C 0-13 python3.8 inference.py --driven_audio andy_say_hello.wav --source_image andy_grove.jpg --result_dir ./result  --cpu --rank=0 --p_num=8 2>&1|tee 1.log &
# numactl -m 0 -C 14-27 python3.8 inference.py --driven_audio andy_say_hello.wav --source_image andy_grove.jpg --result_dir ./result --cpu --rank=1 --p_num=8 2>&1|tee 2.log &

core_num = args.core
unit = core_num / args.slot

with open("run_distributed_infer_{}.sh".format(args.slot), 'w') as f:
    print("run_distributed_infer_{}.sh".format(args.slot))
    for i in range(args.slot):
        socket_idx = 0 if i < args.slot/2 else 1
        start_core = (int)(i * unit)
        end_core = (int)((i+1) * unit -1)
        f.write(f"numactl -m {socket_idx} -C {start_core}-{end_core} python3.8 inference.py --driven_audio {args.driven_audio} --source_image {args.source_image} --result_dir {args.result_dir} --cpu --rank={i} --p_num={args.slot} 2>&1|tee {i}.log &\n")
    f.write("""wait < <(jobs -p)
rm -rf logs
""")

