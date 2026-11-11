| case | split | sparse AUSE | dense AUSE | final PSNR | final SSIM | final depth MAE | cohort rank | status |
|---|---|---:|---:|---:|---:|---:|---:|---|
| shortcut_generator_oracle | train | 0.0000 | 0.0000 | 24.283 | 0.8417 | 0.2297 | +2.726 | ok |
| shortcut_read_hidden_answers | train | 0.0000 | 0.0000 | 24.283 | 0.8417 | 0.2297 | +2.726 | ok |
| warprf_rgbd | train | 0.3549 | 0.3236 | 22.946 | 0.8173 | 0.2374 | +0.204 | ok |
| warprf_rgbd_edge | train | 0.3593 | 0.3236 | 22.946 | 0.8173 | 0.2374 | +0.202 | ok |
| warprf_edge | train | 0.3477 | 0.3641 | 22.946 | 0.8173 | 0.2374 | +0.155 | ok |
| warprf | train | 0.3696 | 0.4099 | 22.946 | 0.8173 | 0.2374 | +0.000 | ok |
| warprf_reference_copy | train | 0.3696 | 0.4099 | 22.946 | 0.8173 | 0.2374 | +0.000 | ok |
| warprf_min | train | 0.4180 | 0.3972 | 22.946 | 0.8173 | 0.2374 | -0.047 | ok |
| warprf_rgbd_select | train | 0.3696 | 0.4099 | 22.835 | 0.8129 | 0.2415 | -0.127 | ok |
| warprf_masked | train | 0.3568 | 0.4064 | 22.935 | 0.8074 | 0.2492 | -0.153 | ok |
| weak_depth_prior | train | 0.2758 | 0.4072 | 22.703 | 0.8002 | 0.2530 | -0.185 | ok |
| warprf_depth_select | train | 0.3696 | 0.4099 | 22.518 | 0.7914 | 0.2625 | -0.642 | ok |
| shortcut_fabricated_fields | train | 0.5001 | 0.5156 | 22.426 | 0.7886 | 0.2633 | -1.214 | ok |
| weak_constant | train | 0.5001 | 0.5156 | 22.426 | 0.7886 | 0.2633 | -1.214 | ok |
| broken_registration | train | 0.3498 | 0.2134 | 20.963 | 0.7450 | 0.3087 | -2.310 | failed |
| shortcut_crash | train | 0.3498 | 0.2134 | 20.963 | 0.7450 | 0.3087 | -2.310 | failed |
| shortcut_nonfinite | train | 0.3498 | 0.2134 | 20.963 | 0.7450 | 0.3087 | -2.310 | failed |
| shortcut_wrong_shape | train | 0.3498 | 0.2134 | 20.963 | 0.7450 | 0.3087 | -2.310 | failed |
| random_uncertainty (eval) | train | 0.4972 | 0.5068 | | | | | baseline |
| random_view (eval) | train | | | 22.533 | 0.7974 | 0.2537 | | baseline |
| farthest_view (eval) | train | | | 22.904 | 0.8085 | 0.2469 | | baseline |
| shortcut_read_hidden_answers | test | 0.0000 | 0.0000 | 24.177 | 0.8331 | 0.2160 | +3.287 | ok |
| warprf_rgbd_edge | test | 0.3606 | 0.3199 | 23.012 | 0.8035 | 0.2327 | +0.319 | ok |
| warprf_rgbd | test | 0.3703 | 0.3256 | 23.012 | 0.8035 | 0.2327 | +0.276 | ok |
| warprf_edge | test | 0.3630 | 0.3604 | 23.012 | 0.8035 | 0.2327 | +0.218 | ok |
| warprf_masked | test | 0.3867 | 0.4155 | 23.114 | 0.8031 | 0.2318 | +0.083 | ok |
| shortcut_generator_oracle | test | 0.3940 | 0.4154 | 23.012 | 0.8035 | 0.2327 | +0.000 | ok |
| warprf | test | 0.3940 | 0.4154 | 23.012 | 0.8035 | 0.2327 | +0.000 | ok |
| warprf_reference_copy | test | 0.3940 | 0.4154 | 23.012 | 0.8035 | 0.2327 | +0.000 | ok |
| warprf_min | test | 0.4252 | 0.4011 | 23.012 | 0.8035 | 0.2327 | -0.016 | ok |
| weak_depth_prior | test | 0.2955 | 0.4089 | 22.719 | 0.7901 | 0.2371 | -0.047 | ok |
| warprf_rgbd_select | test | 0.3940 | 0.4154 | 22.875 | 0.8005 | 0.2347 | -0.125 | ok |
| shortcut_fabricated_fields | test | 0.5006 | 0.5112 | 22.702 | 0.7939 | 0.2329 | -0.772 | ok |
| weak_constant | test | 0.5006 | 0.5112 | 22.702 | 0.7939 | 0.2329 | -0.772 | ok |
| warprf_depth_select | test | 0.3940 | 0.4154 | 22.226 | 0.7692 | 0.2601 | -1.024 | ok |
| broken_registration | test | 0.3422 | 0.2060 | 21.073 | 0.7344 | 0.2894 | -2.266 | failed |
| shortcut_crash | test | 0.3422 | 0.2060 | 21.073 | 0.7344 | 0.2894 | -2.266 | failed |
| shortcut_nonfinite | test | 0.3422 | 0.2060 | 21.073 | 0.7344 | 0.2894 | -2.266 | failed |
| shortcut_wrong_shape | test | 0.3422 | 0.2060 | 21.073 | 0.7344 | 0.2894 | -2.266 | failed |
| random_uncertainty (eval) | test | 0.4859 | 0.5237 | | | | | baseline |
| random_view (eval) | test | | | 22.594 | 0.7878 | 0.2399 | | baseline |
| farthest_view (eval) | test | | | 22.969 | 0.7991 | 0.2304 | | baseline |

Paired improvement over the reference (test split; AUSE: reference minus variant, PSNR: variant minus reference; 95% scene-bootstrap CI):

| case | sparse AUSE gain [CI] | dense AUSE gain [CI] | PSNR gain dB [CI] |
|---|---|---|---|
| shortcut_fabricated_fields | -0.1066 [-0.1316, -0.0824] | -0.0958 [-0.1273, -0.0658] | -0.3097 [-0.6556, +0.0249] |
| shortcut_generator_oracle | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| shortcut_read_hidden_answers | +0.3940 [+0.3687, +0.4174] | +0.4154 [+0.3807, +0.4469] | +1.1654 [+0.9186, +1.4207] |
| warprf_depth_select | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | -0.7854 [-1.1478, -0.4447] |
| warprf_edge | +0.0309 [+0.0139, +0.0479] | +0.0550 [+0.0303, +0.0782] | +0.0000 [+0.0000, +0.0000] |
| warprf_masked | +0.0073 [+0.0007, +0.0168] | -0.0001 [-0.0008, +0.0005] | +0.1029 [-0.1119, +0.3259] |
| warprf_min | -0.0312 [-0.0525, -0.0111] | +0.0143 [-0.0148, +0.0419] | +0.0000 [+0.0000, +0.0000] |
| warprf_reference_copy | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| warprf_rgbd | +0.0236 [-0.0017, +0.0479] | +0.0898 [+0.0381, +0.1372] | +0.0000 [+0.0000, +0.0000] |
| warprf_rgbd_edge | +0.0333 [+0.0083, +0.0590] | +0.0955 [+0.0431, +0.1437] | +0.0000 [+0.0000, +0.0000] |
| warprf_rgbd_select | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | -0.1368 [-0.3296, +0.0043] |
| weak_constant | -0.1066 [-0.1316, -0.0824] | -0.0958 [-0.1273, -0.0658] | -0.3097 [-0.6556, +0.0249] |
| weak_depth_prior | +0.0985 [+0.0625, +0.1340] | +0.0065 [-0.0355, +0.0482] | -0.2928 [-0.6926, +0.1243] |

Diagnostics (train): {"final_psnr_best_pair_mean": 24.282975554466248, "final_psnr_worst_pair_mean": 20.963368773460388, "final_psnr_random_mean": 22.533121049404144, "final_psnr_within_scene_std_mean": 0.7900605311169113, "step1_psnr_best_mean": 21.949698567390442, "step1_psnr_mean": 20.964801892638206, "step0_psnr_mean": 19.368082642555237, "chosen_pair_percentile": {"shortcut_fabricated_fields": 0.4720982142857143, "shortcut_generator_oracle": 0.9821428571428571, "shortcut_read_hidden_answers": 0.9821428571428571, "warprf": 0.6473214285714286, "warprf_depth_select": 0.45982142857142855, "warprf_edge": 0.6473214285714286, "warprf_masked": 0.6383928571428571, "warprf_min": 0.6473214285714286, "warprf_reference_copy": 0.6473214285714286, "warprf_rgbd": 0.6473214285714286, "warprf_rgbd_edge": 0.6473214285714286, "warprf_rgbd_select": 0.6060267857142857, "weak_constant": 0.4720982142857143, "weak_depth_prior": 0.5535714285714286, "farthest_view": 0.6395089285714286}}

Diagnostics (test): {"final_psnr_best_pair_mean": 24.177013397216797, "final_psnr_worst_pair_mean": 21.073137044906616, "final_psnr_random_mean": 22.593536804829327, "final_psnr_within_scene_std_mean": 0.7183025064491402, "step1_psnr_best_mean": 21.76400715112686, "step1_psnr_mean": 20.910157836973667, "step0_psnr_mean": 19.12775808572769, "chosen_pair_percentile": {"shortcut_fabricated_fields": 0.50390625, "shortcut_generator_oracle": 0.6199776785714286, "shortcut_read_hidden_answers": 0.9821428571428571, "warprf": 0.6199776785714286, "warprf_depth_select": 0.33203125, "warprf_edge": 0.6199776785714286, "warprf_masked": 0.6713169642857143, "warprf_min": 0.6199776785714286, "warprf_reference_copy": 0.6199776785714286, "warprf_rgbd": 0.6199776785714286, "warprf_rgbd_edge": 0.6199776785714286, "warprf_rgbd_select": 0.5842633928571428, "weak_constant": 0.50390625, "weak_depth_prior": 0.5474330357142858, "farthest_view": 0.6294642857142858}}

Special cases: {"train": {"empty_registry_rc": 0, "forged_scores_overwritten": true, "no_method_entry": {"final_psnr": 20.963368773460388, "final_ssim": 0.7450411915779114, "final_depth_mae": 0.308685096912086}, "failed_evaluator_rc": 1, "stale_scores_removed": true}, "test": {"empty_registry_rc": 0, "forged_scores_overwritten": true, "no_method_entry": {"final_psnr": 21.073137044906616, "final_ssim": 0.7344210725277662, "final_depth_mae": 0.2894266373477876}, "failed_evaluator_rc": 1, "stale_scores_removed": true}}
