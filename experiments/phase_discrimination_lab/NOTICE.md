# Sources and implementation scope

The baseline/data/loss/evaluation/runtime adapt this repository's R022 harness.
No external paper's implementation source or pretrained weight is bundled.
Third-party dependencies and pretrained weights retain their upstream licenses.

- PSCL: https://www.nature.com/articles/s41598-025-32855-5 ; author code https://github.com/neulmc/PSCL
- BE-Unet/RUCS: https://www.techscience.com/cmc/v87n1/66100/html
- Pixel-wise loss at Al-Si phase interfaces: https://www.nature.com/articles/s41598-019-56008-7
- scSE: https://arxiv.org/abs/1808.08127
- Torchvision ResNet weights: https://pytorch.org/vision/stable/models/resnet.html

D01 is a supervised local contrastive auxiliary objective, not the PSCL
pretraining/fine-tuning framework. D02 is an independently written derived
erosion/dilation residual, not a full reproduction of BE-Unet/RUCS. D03 changes
the contact definition and modality from the tomography paper. Hyperparameters
and expected KoMaP improvements are hypotheses, not reported experimental results.
