# Molecule reconstruction. 

Reconstructs a single molecule from an ECT using filtered backprojection.


# Installation

For managing the dependencies we use `uv`. To initialize the environment run 

```shell
uv sync
```



# Run the reconstruction of the molecule. 

The file `src/datasets/single_molecule.py` contains the hard coded 3D coordinates of the single molecule. 
To run the end to end reconstruction pipeline run the following command in the command line. 

```shell
uv run main.py 
```

# Models 

The latent diffusion model architecture is taken from https://github.com/explainingai-code/StableDiffusion-PyTorch

# Download VGG weights for LPIPS loss.

https://github.com/richzhang/PerceptualSimilarity/blob/master/lpips/weights/v0.1/vgg.pth
