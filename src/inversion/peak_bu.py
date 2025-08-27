def peak_finder_3d(img, width=9):
    resolution = img.shape[-1]
    window_maxima = torch.nn.functional.max_pool3d_with_indices(
        input=img,
        kernel_size=width,
        stride=1,
        padding=width // 2,
    )[1].squeeze()
    out = torch.zeros(size=(len(window_maxima), 30, 3), device=img.device)
    for i, wm in enumerate(window_maxima):
        wm = wm.ravel()
        candidates = torch.unique(wm, sorted=False, return_inverse=False)
        nice_peaks = candidates[(wm[candidates] == candidates).nonzero()]
        res = torch.hstack(
            [
                (nice_peaks // resolution) // resolution,
                (nice_peaks // resolution) % resolution,
                nice_peaks % resolution,
            ]
        )
        out[i, : len(res), :] = res

    return out
