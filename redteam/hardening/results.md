| case | split | sparse AUSE | dense AUSE | final PSNR | final SSIM | final depth MAE | cohort rank | status |
|---|---|---:|---:|---:|---:|---:|---:|---|
| shortcut_generator_oracle | train | 0.0000 | 0.0000 | 24.819 | 0.8534 | 0.3454 | +1.746 | ok |
| warprf_distance_select | train | 0.2786 | 0.3248 | 23.349 | 0.8217 | 0.3552 | +0.096 | ok |
| warprf_masked_select | train | 0.2786 | 0.3248 | 23.458 | 0.8193 | 0.3705 | +0.042 | ok |
| warprf_edge | train | 0.2713 | 0.3028 | 23.303 | 0.8185 | 0.3634 | +0.034 | ok |
| warprf_lookahead_select | train | 0.2786 | 0.3248 | 23.360 | 0.8208 | 0.3700 | +0.008 | ok |
| shortcut_escape | train | 0.2786 | 0.3248 | 23.303 | 0.8185 | 0.3634 | +0.000 | ok |
| warprf | train | 0.2786 | 0.3248 | 23.303 | 0.8185 | 0.3634 | +0.000 | ok |
| warprf_photo_select | train | 0.2786 | 0.3248 | 23.303 | 0.8185 | 0.3634 | +0.000 | ok |
| warprf_reference_copy | train | 0.2786 | 0.3248 | 23.303 | 0.8185 | 0.3634 | +0.000 | ok |
| warprf_min | train | 0.3155 | 0.3101 | 23.303 | 0.8185 | 0.3634 | -0.005 | ok |
| warprf_rgbd_select | train | 0.2786 | 0.3248 | 23.221 | 0.8134 | 0.3593 | -0.057 | ok |
| warprf_negdepth | train | 0.3111 | 0.3713 | 23.303 | 0.8185 | 0.3634 | -0.070 | ok |
| warprf_ensemble_select | train | 0.2786 | 0.3248 | 23.143 | 0.8160 | 0.3606 | -0.084 | ok |
| warprf_neighbourhood_select | train | 0.2786 | 0.3248 | 23.125 | 0.8188 | 0.3692 | -0.121 | ok |
| warprf_rgbd_edge | train | 0.5182 | 0.2656 | 23.303 | 0.8185 | 0.3634 | -0.124 | ok |
| warprf_rgbd | train | 0.6253 | 0.2747 | 23.303 | 0.8185 | 0.3634 | -0.214 | ok |
| neg_depth_prior_hybrid | train | 0.4546 | 0.6532 | 23.303 | 0.8185 | 0.3634 | -0.359 | ok |
| warprf_occlusion_select | train | 0.2786 | 0.3248 | 23.046 | 0.7988 | 0.3882 | -0.428 | ok |
| warprf_depth_select | train | 0.2786 | 0.3248 | 22.693 | 0.7996 | 0.3834 | -0.571 | ok |
| warprf_coverage_select | train | 0.2786 | 0.3248 | 22.480 | 0.7830 | 0.4046 | -0.931 | ok |
| depth_prior_hybrid | train | 1.0774 | 1.0935 | 23.303 | 0.8185 | 0.3634 | -0.994 | ok |
| shortcut_fabricated_fields | train | 0.5888 | 0.6844 | 22.518 | 0.7865 | 0.3970 | -1.339 | ok |
| weak_constant | train | 0.5888 | 0.6844 | 22.518 | 0.7865 | 0.3970 | -1.339 | ok |
| weak_depth_prior | train | 1.0774 | 1.0935 | 22.205 | 0.7864 | 0.3898 | -1.949 | ok |
| broken_registration | train | 1.4916 | 2.0277 | 20.627 | 0.7304 | 0.4858 | -4.801 | failed |
| shortcut_crash | train | 1.4916 | 2.0277 | 20.627 | 0.7304 | 0.4858 | -4.801 | failed |
| shortcut_nonfinite | train | 1.4916 | 2.0277 | 20.627 | 0.7304 | 0.4858 | -4.801 | failed |
| shortcut_wrong_shape | train | 1.4916 | 2.0277 | 20.627 | 0.7304 | 0.4858 | -4.801 | failed |
| random_uncertainty (eval) | train | 0.5871 | 0.6795 | | | | | baseline |
| random_view (eval) | train | | | 22.687 | 0.7974 | 0.3859 | | baseline |
| farthest_view (eval) | train | | | 23.157 | 0.8181 | 0.3647 | | baseline |
| warprf_edge | test | 0.3042 | 0.3151 | 22.923 | 0.8056 | 0.3429 | +0.090 | ok |
| warprf_rgbd_edge | test | 0.4213 | 0.2564 | 22.923 | 0.8056 | 0.3429 | +0.038 | ok |
| warprf_min | test | 0.3742 | 0.3330 | 22.923 | 0.8056 | 0.3429 | +0.006 | ok |
| shortcut_escape | test | 0.3568 | 0.3420 | 22.923 | 0.8056 | 0.3429 | +0.000 | ok |
| shortcut_generator_oracle | test | 0.3568 | 0.3420 | 22.923 | 0.8056 | 0.3429 | +0.000 | ok |
| warprf | test | 0.3568 | 0.3420 | 22.923 | 0.8056 | 0.3429 | +0.000 | ok |
| warprf_reference_copy | test | 0.3568 | 0.3420 | 22.923 | 0.8056 | 0.3429 | +0.000 | ok |
| warprf_photo_select | test | 0.3568 | 0.3420 | 22.902 | 0.8056 | 0.3445 | -0.020 | ok |
| warprf_lookahead_select | test | 0.3568 | 0.3420 | 22.916 | 0.8055 | 0.3482 | -0.033 | ok |
| warprf_rgbd | test | 0.5082 | 0.2712 | 22.923 | 0.8056 | 0.3429 | -0.049 | ok |
| warprf_negdepth | test | 0.3756 | 0.4036 | 22.923 | 0.8056 | 0.3429 | -0.068 | ok |
| warprf_distance_select | test | 0.3568 | 0.3420 | 22.871 | 0.8037 | 0.3523 | -0.094 | ok |
| warprf_masked_select | test | 0.3568 | 0.3420 | 22.810 | 0.7957 | 0.3576 | -0.213 | ok |
| warprf_rgbd_select | test | 0.3568 | 0.3420 | 22.789 | 0.7928 | 0.3676 | -0.301 | ok |
| neg_depth_prior_hybrid | test | 0.4576 | 0.7319 | 22.923 | 0.8056 | 0.3429 | -0.346 | ok |
| warprf_ensemble_select | test | 0.3568 | 0.3420 | 22.678 | 0.7939 | 0.3716 | -0.376 | ok |
| warprf_neighbourhood_select | test | 0.3568 | 0.3420 | 22.560 | 0.7871 | 0.3584 | -0.418 | ok |
| warprf_coverage_select | test | 0.3568 | 0.3420 | 22.343 | 0.7748 | 0.3737 | -0.710 | ok |
| warprf_occlusion_select | test | 0.3568 | 0.3420 | 22.236 | 0.7728 | 0.3757 | -0.794 | ok |
| depth_prior_hybrid | test | 1.0398 | 1.0541 | 22.923 | 0.8056 | 0.3429 | -0.907 | ok |
| warprf_depth_select | test | 0.3568 | 0.3420 | 22.088 | 0.7722 | 0.3895 | -0.955 | ok |
| shortcut_fabricated_fields | test | 0.5877 | 0.7002 | 22.288 | 0.7739 | 0.3876 | -1.272 | ok |
| weak_constant | test | 0.5877 | 0.7002 | 22.288 | 0.7739 | 0.3876 | -1.272 | ok |
| weak_depth_prior | test | 1.0398 | 1.0541 | 22.364 | 0.7816 | 0.3756 | -1.566 | ok |
| broken_registration | test | 1.4869 | 2.1111 | 20.234 | 0.7029 | 0.4801 | -5.095 | failed |
| shortcut_crash | test | 1.4869 | 2.1111 | 20.234 | 0.7029 | 0.4801 | -5.095 | failed |
| shortcut_nonfinite | test | 1.4869 | 2.1111 | 20.234 | 0.7029 | 0.4801 | -5.095 | failed |
| shortcut_wrong_shape | test | 1.4869 | 2.1111 | 20.234 | 0.7029 | 0.4801 | -5.095 | failed |
| random_uncertainty (eval) | test | 0.5855 | 0.7038 | | | | | baseline |
| random_view (eval) | test | | | 22.195 | 0.7723 | 0.3813 | | baseline |
| farthest_view (eval) | test | | | 22.751 | 0.7950 | 0.3703 | | baseline |

Paired improvement over the reference (test split; AUSE: reference minus variant, PSNR: variant minus reference; 95% scene-bootstrap CI):

| case | sparse AUSE gain [CI] | dense AUSE gain [CI] | PSNR gain dB [CI] |
|---|---|---|---|
| depth_prior_hybrid | -0.6830 [-0.7703, -0.6027] | -0.7121 [-0.7888, -0.6310] | +0.0000 [+0.0000, +0.0000] |
| neg_depth_prior_hybrid | -0.1007 [-0.1515, -0.0486] | -0.3899 [-0.4239, -0.3571] | +0.0000 [+0.0000, +0.0000] |
| shortcut_escape | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| shortcut_fabricated_fields | -0.2308 [-0.2735, -0.1848] | -0.3582 [-0.3915, -0.3252] | -0.6346 [-0.9621, -0.3180] |
| shortcut_generator_oracle | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| warprf_coverage_select | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | -0.5798 [-0.8257, -0.3544] |
| warprf_depth_select | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | -0.8347 [-1.1304, -0.5494] |
| warprf_distance_select | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | -0.0517 [-0.2318, +0.1368] |
| warprf_edge | +0.0526 [+0.0205, +0.0893] | +0.0269 [-0.0001, +0.0555] | +0.0000 [+0.0000, +0.0000] |
| warprf_ensemble_select | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | -0.2454 [-0.5019, +0.0023] |
| warprf_lookahead_select | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | -0.0067 [-0.2177, +0.2212] |
| warprf_masked_select | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | -0.1134 [-0.3626, +0.1276] |
| warprf_min | -0.0173 [-0.0454, +0.0136] | +0.0090 [-0.0184, +0.0376] | +0.0000 [+0.0000, +0.0000] |
| warprf_negdepth | -0.0187 [-0.0340, -0.0025] | -0.0616 [-0.0715, -0.0529] | +0.0000 [+0.0000, +0.0000] |
| warprf_neighbourhood_select | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | -0.3629 [-0.5989, -0.1325] |
| warprf_occlusion_select | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | -0.6867 [-1.0031, -0.3509] |
| warprf_photo_select | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | -0.0208 [-0.1171, +0.0531] |
| warprf_reference_copy | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| warprf_rgbd | -0.1514 [-0.1920, -0.1111] | +0.0707 [+0.0231, +0.1211] | +0.0000 [+0.0000, +0.0000] |
| warprf_rgbd_edge | -0.0645 [-0.1028, -0.0247] | +0.0856 [+0.0404, +0.1346] | +0.0000 [+0.0000, +0.0000] |
| warprf_rgbd_select | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | -0.1337 [-0.4007, +0.1202] |
| weak_constant | -0.2308 [-0.2735, -0.1848] | -0.3582 [-0.3915, -0.3252] | -0.6346 [-0.9621, -0.3180] |
| weak_depth_prior | -0.6830 [-0.7703, -0.6027] | -0.7121 [-0.7888, -0.6310] | -0.5590 [-0.8475, -0.2811] |

Paired improvement over the reference (train split; AUSE: reference minus variant, PSNR: variant minus reference; 95% scene-bootstrap CI):

| case | sparse AUSE gain [CI] | dense AUSE gain [CI] | PSNR gain dB [CI] |
|---|---|---|---|
| depth_prior_hybrid | -0.7988 [-0.9725, -0.6475] | -0.7687 [-0.9250, -0.6111] | +0.0000 [+0.0000, +0.0000] |
| neg_depth_prior_hybrid | -0.1760 [-0.2645, -0.0943] | -0.3284 [-0.3763, -0.2813] | +0.0000 [+0.0000, +0.0000] |
| shortcut_escape | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| shortcut_fabricated_fields | -0.3102 [-0.3693, -0.2556] | -0.3596 [-0.4049, -0.3121] | -0.7851 [-1.1632, -0.4056] |
| shortcut_generator_oracle | +0.2786 [+0.2151, +0.3428] | +0.3248 [+0.2673, +0.3783] | +1.5156 [+1.2001, +1.8453] |
| warprf_coverage_select | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | -0.8231 [-1.3436, -0.3615] |
| warprf_depth_select | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | -0.6107 [-1.2927, +0.0698] |
| warprf_distance_select | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0456 [-0.2512, +0.3734] |
| warprf_edge | +0.0073 [-0.0386, +0.0519] | +0.0221 [-0.0170, +0.0562] | +0.0000 [+0.0000, +0.0000] |
| warprf_ensemble_select | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | -0.1604 [-0.5380, +0.2357] |
| warprf_lookahead_select | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0564 [-0.1552, +0.2890] |
| warprf_masked_select | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.1551 [-0.1739, +0.4421] |
| warprf_min | -0.0369 [-0.0598, -0.0084] | +0.0147 [-0.0212, +0.0465] | +0.0000 [+0.0000, +0.0000] |
| warprf_negdepth | -0.0325 [-0.0513, -0.0136] | -0.0465 [-0.0607, -0.0295] | +0.0000 [+0.0000, +0.0000] |
| warprf_neighbourhood_select | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | -0.1788 [-0.5721, +0.2257] |
| warprf_occlusion_select | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | -0.2574 [-0.8191, +0.4008] |
| warprf_photo_select | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| warprf_reference_copy | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| warprf_rgbd | -0.3467 [-0.4538, -0.2439] | +0.0501 [-0.0194, +0.1167] | +0.0000 [+0.0000, +0.0000] |
| warprf_rgbd_edge | -0.2396 [-0.3371, -0.1521] | +0.0592 [-0.0085, +0.1244] | +0.0000 [+0.0000, +0.0000] |
| warprf_rgbd_select | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | -0.0820 [-0.5096, +0.3406] |
| weak_constant | -0.3102 [-0.3693, -0.2556] | -0.3596 [-0.4049, -0.3121] | -0.7851 [-1.1632, -0.4056] |
| weak_depth_prior | -0.7988 [-0.9725, -0.6475] | -0.7687 [-0.9250, -0.6111] | -1.0984 [-1.5930, -0.5890] |

Reference vs baselines and oracles (train; mean [95% CI], scenes better/worse):

- depth_uncertainty_sparse: reference AUSE gain over random_uncertainty: +0.3084 [+0.2474, +0.3662] (16/0)
- depth_uncertainty_dense: reference AUSE gain over random_uncertainty: +0.3547 [+0.3078, +0.3976] (16/0)
- active: reference PSNR gain over random_view: +0.6161 [+0.2994, +0.9186] (14/2)
- active: reference PSNR gain over farthest_view: +0.1460 [-0.1621, +0.4410] (9/5)
- active: oracle_uncertainty PSNR gain over reference: +0.3831 [+0.0026, +0.7801] (10/4)
- active: greedy_oracle PSNR gain over reference: +1.3412 [+0.9730, +1.7239] (15/0)
- active: best_pair PSNR gain over reference: +1.5156 [+1.1885, +1.8492] (16/0)

Diagnostics (train, isolation=namespace): {"final_psnr_best_pair_mean": 24.81902587413788, "final_psnr_worst_pair_mean": 20.626997411251068, "final_psnr_random_mean": 22.687332497537135, "final_psnr_oracle_uncertainty_mean": 23.68654775619507, "final_psnr_greedy_oracle_mean": 24.644641399383545, "final_psnr_within_scene_std_mean": 0.8525853069134959, "step1_psnr_best_mean": 22.119486451148987, "step1_psnr_mean": 21.001883629709482, "step0_psnr_mean": 19.234079003334045, "chosen_pair_percentile": {"depth_prior_hybrid": 0.7286458333333333, "neg_depth_prior_hybrid": 0.7286458333333333, "shortcut_escape": 0.7286458333333333, "shortcut_fabricated_fields": 0.4265625, "shortcut_generator_oracle": 0.9958333333333333, "warprf": 0.7286458333333333, "warprf_coverage_select": 0.42708333333333337, "warprf_depth_select": 0.509375, "warprf_distance_select": 0.7333333333333334, "warprf_edge": 0.7286458333333333, "warprf_ensemble_select": 0.6630208333333334, "warprf_lookahead_select": 0.76484375, "warprf_masked_select": 0.75, "warprf_min": 0.7286458333333333, "warprf_negdepth": 0.7286458333333333, "warprf_neighbourhood_select": 0.65234375, "warprf_occlusion_select": 0.6052083333333333, "warprf_photo_select": 0.7286458333333333, "warprf_reference_copy": 0.7286458333333333, "warprf_rgbd": 0.7286458333333333, "warprf_rgbd_edge": 0.7286458333333333, "warprf_rgbd_select": 0.6533854166666667, "weak_constant": 0.4265625, "weak_depth_prior": 0.384375, "farthest_view": 0.67734375}}

Reference vs baselines and oracles (test; mean [95% CI], scenes better/worse):

- depth_uncertainty_sparse: reference AUSE gain over random_uncertainty: +0.2287 [+0.1813, +0.2727] (38/2)
- depth_uncertainty_dense: reference AUSE gain over random_uncertainty: +0.3618 [+0.3260, +0.3961] (40/0)
- active: reference PSNR gain over random_view: +0.7282 [+0.5071, +0.9579] (35/5)
- active: reference PSNR gain over farthest_view: +0.1720 [-0.0480, +0.4247] (21/17)
- active: oracle_uncertainty PSNR gain over reference: +0.0906 [-0.1438, +0.3001] (20/14)
- active: greedy_oracle PSNR gain over reference: +0.9163 [+0.7341, +1.1074] (39/1)
- active: best_pair PSNR gain over reference: +1.1539 [+0.9732, +1.3402] (40/0)

Diagnostics (test, isolation=namespace): {"final_psnr_best_pair_mean": 24.076817750930786, "final_psnr_worst_pair_mean": 20.23383524417877, "final_psnr_random_mean": 22.194748235146204, "final_psnr_oracle_uncertainty_mean": 23.013522958755495, "final_psnr_greedy_oracle_mean": 23.839256620407106, "final_psnr_within_scene_std_mean": 0.7708008928060872, "step1_psnr_best_mean": 21.45809450149536, "step1_psnr_mean": 20.47861324697733, "step0_psnr_mean": 18.70665488243103, "chosen_pair_percentile": {"depth_prior_hybrid": 0.7440625, "neg_depth_prior_hybrid": 0.7440625, "shortcut_escape": 0.7440625, "shortcut_fabricated_fields": 0.4998958333333333, "shortcut_generator_oracle": 0.7440625, "warprf": 0.7440625, "warprf_coverage_select": 0.54375, "warprf_depth_select": 0.4761458333333334, "warprf_distance_select": 0.7479166666666667, "warprf_edge": 0.7440625, "warprf_ensemble_select": 0.6742708333333333, "warprf_lookahead_select": 0.7510416666666666, "warprf_masked_select": 0.6966666666666667, "warprf_min": 0.7440625, "warprf_negdepth": 0.7440625, "warprf_neighbourhood_select": 0.6177083333333334, "warprf_occlusion_select": 0.525, "warprf_photo_select": 0.748125, "warprf_reference_copy": 0.7440625, "warprf_rgbd": 0.7440625, "warprf_rgbd_edge": 0.7440625, "warprf_rgbd_select": 0.7063541666666666, "weak_constant": 0.4998958333333333, "weak_depth_prior": 0.55875, "farthest_view": 0.6942708333333334}}

Escape routes taken by shortcut_escape (None = no route worked): {"train": {"None": 32}, "train_example_attempts": {"direct": "FileNotFoundError", "chmod": "OSError", "proc_root": "no readable path via 1 visible pids", "git": "fatal: not a git repository (or any parent up to mount point /home)\nStopping at filesystem boundary (GIT_DISCOVERY_ACROS"}, "test": {"None": 80}, "test_example_attempts": {"direct": "FileNotFoundError", "chmod": "OSError", "proc_root": "no readable path via 1 visible pids", "git": "fatal: not a git repository (or any parent up to mount point /home)\nStopping at filesystem boundary (GIT_DISCOVERY_ACROS"}}

Special cases: {"train": {"empty_registry_rc": 0, "forged_scores_overwritten": true, "no_method_entry": {"final_psnr": 20.626997411251068, "final_ssim": 0.7303725853562355, "final_depth_mae": 0.4857948822900653}, "failed_evaluator_rc": 1, "stale_scores_removed": true}, "test": {"empty_registry_rc": 0, "forged_scores_overwritten": true, "no_method_entry": {"final_psnr": 20.23383524417877, "final_ssim": 0.7028542436659336, "final_depth_mae": 0.48013468459248543}, "failed_evaluator_rc": 1, "stale_scores_removed": true}}
