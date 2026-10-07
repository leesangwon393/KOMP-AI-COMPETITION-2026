# U²-Net source attribution and modifications

Upstream project: [xuebinqin/U-2-Net](https://github.com/xuebinqin/U-2-Net).
Paper: [Qin et al., U²-Net: Going Deeper with Nested U-Structure for Salient Object Detection](https://arxiv.org/abs/2005.09007).

`komap_phase/vendor_u2net.py` is an unchanged snapshot of the upstream
[`model/u2net.py`](https://raw.githubusercontent.com/xuebinqin/U-2-Net/master/model/u2net.py).
The upstream Apache License, Version 2.0 is included in `LICENSE_U2NET.txt`.
File digests and the retrieval source are recorded in
`docs/U2NET_SOURCE_PROVENANCE.json`; all release files also appear in
`SHA256SUMS.txt` inside the ZIP.

Project adaptations live outside the vendor snapshot, in `rsu_models.py`,
`models.py`, `objectives.py`, and the training/evaluation pipeline:

- Select encoder stages2–6 and adapt their channels to the existing B decoder.
- Convert the full model to four-class raw logits, removing binary sigmoid.
- Add optional scSE after decoder RSUs and six-side normalized auxiliary loss.
- Apply the existing grayscale normalization, central224 supervision inside
  context448, RGB-mask decoding, contact-weighted CE and single/D4 evaluation.
- Add isolated patch-sampling RNG and checkpoint resume state for matched
  sampling across different model initializations.

The upstream paper/code targets binary salient-object detection. These KoMaP
multiclass, backbone-adapter, attention, loss and evaluation configurations are
new adaptations; they are not a reproduction of the paper's reported scores.
No upstream binary trained weights are included or used.
