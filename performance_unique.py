import argparse
import time

import torch


def test_unique(use_cuda, use_dim):
    N = int(1e7)
    device = "cuda:0" if use_cuda else "cpu"
    for high in torch.logspace(1, 5, 5):
        high = int(high.item())
        print("Timing unique with high={} for {} values on {}".format(high, N, device))
        x = torch.randint(low=0, high=high, size=(N,))
        if use_cuda:
            x = x.to(device)

        torch.cuda.synchronize()
        start = time.time()
        if use_dim:
            unique, inv = torch.unique(x, sorted=False, return_inverse=True, dim=0)
        else:
            unique, inv = torch.unique(x, sorted=False, return_inverse=True)
        torch.cuda.synchronize()
        print(time.time() - start)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_cuda", action="store_true", default=False)
    parser.add_argument("--use_dim", action="store_true", default=False)
    args = parser.parse_args()
    use_cuda = args.use_cuda and torch.cuda.is_available()
    use_dim = args.use_dim
    test_unique(use_cuda=use_cuda, use_dim=use_dim)


if __name__ == "__main__":
    main()
