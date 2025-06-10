# COMP 484 — Pix2Pix: Sketch → Photo

Conditional GAN for image-to-image translation on Edges2Shoes dataset.
Kathmandu University · 2025

## Architecture

- **Generator**: U-Net (256→256), 8-layer encoder–decoder + skip connections
- **Discriminator**: PatchGAN 70×70
- **Loss**: Adversarial BCE + L1 (λ=100)

## Training

```bash
cd train && pip install -r requirements.txt
python train.py --dataset ./edges2shoes --epochs 50
```

## Inference

```bash
cd inference_app && pip install -r requirements.txt
python app.py
```

More details coming as training results land.

---

## Colab Notebook

The original training notebook (`notebook/pix2pix_colab.ipynb`) runs on Google Colab with Drive mounting for persistent checkpoints. It is functionally equivalent to `train/train.py` but structured as cells for interactive experimentation. To use the trained model locally, download `pix2pix_generator.keras` from Drive and load it in the inference app.



