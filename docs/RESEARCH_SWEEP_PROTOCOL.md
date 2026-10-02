**Paper-inspired independent ablations — protocol fixed before training**

Sequence: pixel contrast → snake branch → boundary auxiliary head → clDice.
Each starts a fresh ImageNet ResNet34 + scSE U-Net; methods are not stacked.
Reference: runs/resnet34_unet_scse_ce_dice (single mIoU .7767221; D4 .7802636).

Common settings: seed 42, 100 epochs, batch 4, 420 source-balanced patches/epoch,
224 crop, stride 112, 50% enriched sampling (Eutectic >=10%), random D4 augmentation,
AdamW lr 3e-4, weight decay 1e-4, cosine T_max=100. Main loss .5 CE + .5 weighted
Dice with class weights [1,1,2,1]. All encoder weights trainable. No AMP.

Common initial model weights are generated before optional modules. Initialization
of optional modules uses fork_rng; contrastive anchor selection uses a separate
CPU generator. The crop RNG is therefore unaffected by extra modules/losses.
CPU checks verified equal common weights, initial outputs, and global RNG state.

Pixel contrast: ICCV 2021 Cross-Image Pixel Contrast-inspired supervised contrastive
loss on sampled full-resolution decoder features. 32 hard/easy interior pixels per
class per image, 32-D projected embeddings, temperature .1, cross-source positives,
all other sampled pixels except self in the denominator. No memory bank. Coefficient
.05. This is an adaptation, not identical to the author's exact objective.
Source: https://openaccess.thecvf.com/content/ICCV2021/papers/Wang_Exploring_Cross-Image_Pixel_Contrast_for_Semantic_Segmentation_ICCV_2021_paper.pdf

DSConv: ICCV 2023-inspired 5-point horizontal and vertical snakes on the 112×112
decoder feature. Cumulative perpendicular tanh offsets, zero initial offsets,
bilinear grid sampling, 16-channel branch then fusion back to 64 channels.
The normal path is retained with a zero-initialized learnable residual gain.
No persistent-homology loss or full DSCNet architecture. MPS lacks grid_sample
backward in the installed PyTorch version. A border-clamped four-neighbor bilinear
gather implementation is used instead; CPU output and feature/coordinate gradients
were checked against grid_sample with border padding.
Source: https://github.com/YaoleiQi/DSCNet

Boundary: four class-specific boundary maps derived solely from Train masks with
a 3×3 morphological band. Small auxiliary head on the full-resolution decoder,
positive-weighted BCE (weight clamped to [1,10]), coefficient .1. Head unused during
inference. Inspired by shape supervision; does not reproduce Gated-SCNN's gated stream.
Source: https://openaccess.thecvf.com/content_ICCV_2019/html/Takikawa_Gated-SCNN_Gated_Shape_CNNs_for_Semantic_Segmentation_ICCV_2019_paper.html

clDice: Eutectic one-vs-rest soft skeleton loss, 8 erosion iterations, coefficient
.05. Skip absent-GT samples. No artificial single-component constraint. Train GT
morphology and baseline Valid-mask skeleton coverage are recorded in topology_audit.json.
The audit is a proxy diagnosis, not a proof of true connectivity.
Source: https://openaccess.thecvf.com/content/CVPR2021/html/Shit_clDice_-_A_Novel_Topology-Preserving_Loss_Function_for_Tubular_Structure_CVPR_2021_paper.html

All auxiliary losses: disabled for epochs 1–10; linear coefficient ramp at 11–20;
full coefficient after epoch 20. Snake architectural branch is active from epoch 1.
No validation hyperparameter tuning. Fixed 100 epochs for each method.

Select the best checkpoint by single-view original-resolution Validation mIoU,
then perform one identical D4 evaluation (inverse transforms, probability averaging,
overlap tile averaging). Keep historical metric absent=one, report alternative
absent policies as well. Never sample from Validation or Test for training.

Save per-epoch per-image metrics, best/last checkpoints, RNG/optimizer/scheduler
state, dataset hashes, source snapshot, masks and D4 metrics. Report whole mIoU,
Al3Ni/Eutectic IoU, paired per-image deltas and image-bootstrap intervals. Intervals
do not account for seed variation or repeated best-checkpoint selection bias.

Early stopping review: retrospective min_delta=.001, minimum epoch=40. Patience 15
would lose .6088 pp on the historical scSE run. Patience 20 and 30 would both miss
the historical MIMU CE/Dice maximum by .5928 pp. For this sweep, keep the prespecified
100 epochs and report retrospective stopping simulations, not actual early stopping.

Execution: python3 research_sweep.py --device mps
Results: runs/research_sweep_v1/RESULTS.md and summary.json (updated after each method).
Interrupted experiments resume from last.pt. The existing completed runs are untouched.
