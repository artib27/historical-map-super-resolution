# Super-Resolution of Historical Maps using ESRGAN

## Objective

This project explores **4× image super-resolution** for historical map scans. It compares bicubic interpolation, a pretrained ESRGAN model, and a model trained on historical map patches. The goal is to improve the visibility of map lines and labels and compare the methods through images and quality metrics. Generated details are visual estimates, not historically verified information.

## Project files

The main project record is [`DeepLearning_Project_ESRGAN.ipynb`](DeepLearning_Project_ESRGAN.ipynb). Its [HTML export](DeepLearning_Project_ESRGAN.html) shows the completed notebook. The local [`ESRGAN/`](ESRGAN/) directory contains upstream model code. Download the pretrained checkpoint separately from the source linked below.

| Location | Contents |
| --- | --- |
| [`files/maps_hr/maps_hr/`](files/maps_hr/maps_hr/) | Three high-resolution map scans and their text metadata. The notebook attributes the maps to the David Rumsey Historical Map Collection. |
| [`files/maps_test/`](files/maps_test/) | Three test map images and a saved pretrained-model example. |
| [`files/hr_patches/`](files/hr_patches/) and [`files/lr_patches/`](files/lr_patches/) | 13,192 paired high- and low-resolution training patches. |
| [`RRDB_ESRGAN_x4.pth` download page](https://huggingface.co/databuzzword/esrgan/blob/main/RRDB_ESRGAN_x4.pth) | Pretrained 4× ESRGAN weights used by the notebook. Download the file and place it at `ESRGAN/models/RRDB_ESRGAN_x4.pth` locally; the checkpoint is excluded from this Git repository. |
| [`files/trained_generator.pth`](files/trained_generator.pth) | Saved weights of the project-trained generator. |
| [`files/outputs/`](files/outputs/) | Individual results and combined comparisons for the three test maps. |

## Workflow

1. **Prepare paired patches.** The notebook crops the high-resolution maps into non-overlapping **128 × 128** patches and downsamples each one by 4× with bicubic interpolation to make a **32 × 32** input. The introduction mentions blur, but the active patch-generation code uses bicubic downsampling only.
2. **Create a baseline.** Bicubic interpolation enlarges each test image by 4×.
3. **Apply pretrained ESRGAN.** A pretrained 4× residual-in-residual dense block (RRDB) model produces a second enlarged result.
4. **Train a project model.** The notebook uses an 80/20 random split of patches, a 23-block RRDB generator, and a discriminator with spectral normalization, self-attention, and a minibatch standard-deviation channel. The training code specifies **15 epochs**, batch size **16**, Adam at **`1e-4`**, and a weighted combination of L1 pixel, VGG perceptual, and adversarial losses.
5. **Compare outputs.** For each test map, the notebook saves the original, bicubic, pretrained, and project-trained images, plus a four-panel comparison. It calculates average PSNR and SSIM (higher is better) and LPIPS (lower is better).

## Final output and results

The [`files/outputs/`](files/outputs/) folder contains 15 images: each of `map_test_1`, `map_test_2`, and `map_test_3` has `_original.png`, `_bicubic.png`, `_sr_pretrained.png`, `_sr_trained.png`, and `_comparison.png` versions. See the [first combined comparison](files/outputs/map_test_1_comparison.png) for an example.

The **executed notebook output** reports these averages across the three test images:

| Method | PSNR (dB) ↑ | SSIM ↑ | LPIPS ↓ |
| --- | ---: | ---: | ---: |
| Bicubic | 33.7 | 0.97 | 0.07 |
| Pretrained ESRGAN | **37.1** | **0.99** | **0.01** |
| Project-trained model | 11.7 | 0.71 | 0.25 |

The pretrained model has the best reported scores. The project-trained result shows visible artifacts and color shifts. A later Markdown table in the notebook lists **0.77** for the trained model's SSIM; **0.71** is the value printed by the executed evaluation cell.

## Running the notebook

Use a Python 3 Jupyter environment with PyTorch, torchvision, Pillow, NumPy, pandas, Matplotlib, scikit-image, and `lpips`. The notebook also includes setup cells that use Git, Git LFS, `wget`, `gdown`, and `opencv-python` to obtain upstream code, model weights, and map archives.

The notebook was written for AWS or Colab and sets `ROOT_DIR` to an absolute path. Before a local run, point `ROOT_DIR` to this repository's `files/` directory and start Jupyter from the repository root. Download `RRDB_ESRGAN_x4.pth` from the link above into `ESRGAN/models/`, then adjust or skip clone and download cells when using the local maps and ESRGAN code. Its training code requires CUDA as written because the perceptual-loss class explicitly moves tensors to the GPU. To inspect the completed work without retraining, open the HTML export and saved output images.

## Evaluation limits

- The notebook resizes each 4× output back to the original test image's dimensions before calculating PSNR, SSIM, and LPIPS. These scores measure similarity to the input image, **not** accuracy against an independent high-resolution reference.
- Randomly splitting patches from the same small group of maps does not measure generalization to entirely unseen maps.
- The test-image path normalizes inputs differently for the pretrained and project-trained models and clamps the trained model's output to `[0, 1]`. Review that preprocessing before using the checkpoint for a new benchmark.

The notebook suggests expanding the map dataset, simulating damage such as tears and stains, and tuning training hyperparameters as future work.
