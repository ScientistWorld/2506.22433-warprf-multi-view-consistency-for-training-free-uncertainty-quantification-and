# WarpRF: Multi-View Consistency for Training-Free Uncertainty Quantification and Applications in Radiance Fields

| This paper | Scientist World, the framework this environment belongs to |
|:---:|:---:|
| <a href="https://arxiv.org/abs/2506.22433"><img alt="This paper on arXiv" src="https://img.shields.io/badge/arXiv-2506.22433-b31b1b?logo=arxiv&logoColor=white"></a> | <a href="https://arxiv.org/abs/2611.08478"><img alt="Scientist World on arXiv" src="https://img.shields.io/badge/arXiv-2611.08478-b31b1b?logo=arxiv&logoColor=white"></a> <a href="https://github.com/zlab-princeton/scientist_world"><img alt="Scientist World code" src="https://img.shields.io/badge/Code-Scientist_World-181717?logo=github&logoColor=white"></a> <a href="https://github.com/zlab-princeton/scientist_world_cli"><img alt="Command line code" src="https://img.shields.io/badge/Code-Command_line-181717?logo=github&logoColor=white"></a> |

| | |
|---|---|
| **Paper ID** | [2506.22433](https://arxiv.org/abs/2506.22433) |
| **Domain** | Computer Vision / Neural Rendering / Active View Selection |
| **TL;DR** | WarpRF estimates radiance-field uncertainty from cross-view rendering consistency and uses it to select informative new views without training an uncertainty model. |

## Summary

The paper studies uncertainty estimation for radiance fields and its use in active view selection and active mapping. It proposes a training-free method that compares target renderings with source-view renderings warped into the target camera. The reported experiments show improved AUSE over prior uncertainty methods and improved downstream view selection for 3DGS and SVRaster, plus active mapping gains for NeRF and 3DGS.

## Problem

How can a radiance-field system estimate which rendered pixels, depths, or unseen camera views are unreliable, and use that uncertainty to choose additional training views that improve novel-view synthesis and 3D reconstruction? The benchmark should evaluate uncertainty estimates for already trained radiance fields and downstream active data acquisition under limited-view settings.

## The paper's method

WarpRF introduces training-free uncertainty quantification for radiance fields using multi-view consistency. It treats rendered training-view images and depths as reliable references, warps them into an unseen target view using camera geometry and rendered depth, then measures disagreement with the target rendering as uncertainty.

## Key results of the paper

- On Table 1 AUSE, 3DGS + WarpRF reports 0.337 on ScanNet++ and 0.227 on ETH3D, better than 3DGS + FisherRF at 0.355 and 0.295.
- On active view selection for Mip-NeRF360, 3DGS + WarpRF reports 20.715 PSNR, 0.6174 SSIM, and 0.3487 LPIPS.
- On SVRaster active view selection, WarpRF improves over random and farthest-view baselines on NeRF Synthetic with 20 views and Tanks and Temples.

## Leaderboard

FNI of AI scientists on this paper, grader-verified. 0 matches the paper's method, higher is better.

| Model | Innovate |
|---|---:|
| `gpt-5.6-luna-xhigh` | -33.1 |
| `gpt-5.6-sol-xhigh` | -64.0 |
| `gpt-5.6-terra-xhigh` | -167.3 |

`innovate`: the model designs its own method. `improve`: it starts from the paper's method.

## Run it

```bash
pip install sciworld
sw research 2506.22433 --mode innovate --agent claude-code
sw evaluate 2506.22433
```

## More

The full problem statement, method notes, evaluation protocol and reproduction log are in [DETAILS.md](DETAILS.md).

