import torch

x_orig_list = torch.load("orig_pts_list.pt")
x_recon_list = torch.load("pts_list.pt")


# |%%--%%| <UtYgAUjRu7|EbIFFaVbyv>

correct = 0
total = 0
too_few = 0
too_many = 0

for i, (x_recon, x_orig) in enumerate(zip(x_recon_list, x_orig_list)):
    total += 1

    if x_recon.shape[0] == x_orig.shape[0]:
        correct += 1
        print(i, x_recon.shape[0], x_orig.shape[0])

    if x_recon.shape[0] < x_orig.shape[0]:
        too_few += 1
        print(i, x_recon.shape[0], x_orig.shape[0])

    if x_recon.shape[0] > x_orig.shape[0]:
        too_many += 1

    if i == 1000:
        break

print("accuracy", correct / total)
print("too_few", too_few / total)
print("too_many", too_many / total)
