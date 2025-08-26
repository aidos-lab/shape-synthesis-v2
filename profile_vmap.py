import torch
from torch.profiler import ProfilerActivity, profile, record_function

input1 = torch.randn(3, 3, device="cpu")
input2 = torch.randn(3, 3, device="cpu")

with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
) as prof:
    input1 = input1.to("cuda")
    input2 = input2.to("cuda")
    output1 = input1 + 1.0
    output2 = input2 + 2.0
    output = output1 + output2
print(prof.key_averages().table(sort_by="cuda_time_total"))

# import functools
#
# import torch
# from torch.profiler import ProfilerActivity, profile
#
# torch.set_float32_matmul_precision("medium")
# batched_dot = torch.vmap(torch.dot, in_dims=(0, None))  # [N, D], [D] -> [N]
#
# N = 1000
#
# x, y = torch.randn(2000, N).cuda(), torch.randn(N).cuda()
#
# with torch.profiler.profile(
#     activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU],
#     # schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
#     # on_trace_ready=torch.profiler.tensorboard_trace_handler("./log/batched_dot"),
#     # record_shapes=True,
#     # profile_memory=True,
#     # with_stack=True,
# ) as prof:
#     x @ y
#
# print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
#
#

# import torch
# from torch.profiler import ProfilerActivity, profile, record_function
#
# with profile(
#     activities=[torch.profiler.ProfilerActivity.CUDA],
#     schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
#     on_trace_ready=torch.profiler.tensorboard_trace_handler("./log/model"),
#     profile_memory=False,
#     record_shapes=True,
#     with_stack=True,
# ) as prof:
#     for _ in range(10):
#         y = torch.randn(1).cuda() + torch.randn(1).cuda()
#         prof.step()
#
#
# print(prof.key_averages())
