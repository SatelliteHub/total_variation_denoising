# Total Variation Denoising
Image processing: total-variation denoising

## Run the experiments

From the project folder, choose the parameters using the validation images:

```bash
python run_experiments.py --tune-validation
```

The script compares three fixed weights and three adaptive scale factors. It saves the per-image scores to `outputs/validation_tuning.csv`, the selected values and their validation scores to `outputs/selected_parameters.json`, and a paired validation comparison to `outputs/validation_alpha-..._weight-....csv`.

Then run the independent test images using the selected values:

```bash
python run_experiments.py --split test --config outputs/selected_parameters.json
```

Test seeds are separate from development and validation seeds. Keep the test results for the final comparison rather than changing parameters after viewing them.

The runner reports MSE and PSNR for the noisy, fixed-weight, and adaptive images. Lower MSE and higher PSNR mean the result is closer to the clean image.

## Run local checks

From the project folder, run:

```bash
python -m unittest discover -s tests -v
```

These checks use a small hand-worked gradient and divergence example, check the negative-adjoint relationship, confirm constant-image preservation, and check the output shape, finite values and input preservation for both denoisers.

## Textured stress case

Run an additional checkerboard stress case with the parameters selected on validation:

```bash
python run_texture_stress.py
```

This uses 20 new seeds, starting at 3000, and a noise standard deviation of 0.10. It saves paired whole-image and checkerboard-region measurements to `outputs/texture_stress_results.csv` and an example image with a zoomed texture patch to `outputs/texture_stress_example.png`. The texture contrast and correlation values show how much of the clean checkerboard remains in each reconstruction. Do not use these stress results to retune the parameters.
