| case | split | sparse AUSE | dense AUSE | final PSNR | final SSIM | final depth MAE | cohort rank | status |
|---|---|---:|---:|---:|---:|---:|---:|---|
| shortcut_generator_oracle | train | 0.0000 | 0.0000 | 24.632 | 0.8499 | 0.3515 | +1.350 | ok |
| warprf_lookahead_select | train | 0.2821 | 0.3242 | 23.652 | 0.8315 | 0.3557 | +0.078 | ok |
| warprf_distance_select | train | 0.2821 | 0.3242 | 23.692 | 0.8301 | 0.3604 | +0.060 | ok |
| warprf_edge | train | 0.2728 | 0.3100 | 23.559 | 0.8288 | 0.3575 | +0.031 | ok |
| shortcut_escape | train | 0.2821 | 0.3242 | 23.559 | 0.8288 | 0.3575 | +0.000 | ok |
| warprf | train | 0.2821 | 0.3242 | 23.559 | 0.8288 | 0.3575 | +0.000 | ok |
| warprf_photo_select | train | 0.2821 | 0.3242 | 23.559 | 0.8288 | 0.3575 | +0.000 | ok |
| warprf_reference_copy | train | 0.2821 | 0.3242 | 23.559 | 0.8288 | 0.3575 | +0.000 | ok |
| warprf_min | train | 0.3140 | 0.3177 | 23.559 | 0.8288 | 0.3575 | -0.004 | ok |
| warprf_masked_select | train | 0.2821 | 0.3242 | 23.601 | 0.8263 | 0.3666 | -0.052 | ok |
| warprf_negdepth | train | 0.3159 | 0.3702 | 23.559 | 0.8288 | 0.3575 | -0.069 | ok |
| warprf_rgbd_select | train | 0.2821 | 0.3242 | 23.455 | 0.8210 | 0.3501 | -0.070 | ok |
| warprf_rgbd_edge | train | 0.5083 | 0.2590 | 23.559 | 0.8288 | 0.3575 | -0.107 | ok |
| warprf_ensemble_select | train | 0.2821 | 0.3242 | 23.375 | 0.8227 | 0.3575 | -0.140 | ok |
| warprf_rgbd | train | 0.6183 | 0.2685 | 23.559 | 0.8288 | 0.3575 | -0.199 | ok |
| neg_depth_prior_hybrid | train | 0.4577 | 0.6517 | 23.559 | 0.8288 | 0.3575 | -0.356 | ok |
| warprf_neighbourhood_select | train | 0.2821 | 0.3242 | 23.055 | 0.8144 | 0.3734 | -0.457 | ok |
| warprf_depth_select | train | 0.2821 | 0.3242 | 22.804 | 0.8048 | 0.3704 | -0.639 | ok |
| warprf_occlusion_select | train | 0.2821 | 0.3242 | 22.928 | 0.8033 | 0.3901 | -0.704 | ok |
| warprf_coverage_select | train | 0.2821 | 0.3242 | 22.772 | 0.7918 | 0.4029 | -0.944 | ok |
| depth_prior_hybrid | train | 1.0781 | 1.0948 | 23.559 | 0.8288 | 0.3575 | -0.990 | ok |
| shortcut_fabricated_fields | train | 0.5896 | 0.6833 | 22.670 | 0.7938 | 0.3875 | -1.386 | ok |
| weak_constant | train | 0.5896 | 0.6833 | 22.670 | 0.7938 | 0.3875 | -1.386 | ok |
| weak_depth_prior | train | 1.0781 | 1.0948 | 22.341 | 0.7915 | 0.3875 | -2.064 | ok |
| broken_registration | train | 1.4950 | 2.0367 | 20.816 | 0.7363 | 0.4828 | -4.873 | failed |
| shortcut_crash | train | 1.4950 | 2.0367 | 20.816 | 0.7363 | 0.4828 | -4.873 | failed |
| shortcut_nonfinite | train | 1.4950 | 2.0367 | 20.816 | 0.7363 | 0.4828 | -4.873 | failed |
| shortcut_wrong_shape | train | 1.4950 | 2.0367 | 20.816 | 0.7363 | 0.4828 | -4.873 | failed |
| random_uncertainty (eval) | train | 0.5864 | 0.6813 | | | | | baseline |
| random_view (eval) | train | | | 22.774 | 0.8011 | 0.3821 | | baseline |
| farthest_view (eval) | train | | | 23.459 | 0.8279 | 0.3485 | | baseline |
| warprf_edge | test | 0.3026 | 0.3190 | 23.267 | 0.8165 | 0.3352 | +0.093 | ok |
| warprf_lookahead_select | test | 0.3484 | 0.3580 | 23.278 | 0.8185 | 0.3326 | +0.035 | ok |
| warprf_rgbd_edge | test | 0.4384 | 0.2644 | 23.267 | 0.8165 | 0.3352 | +0.025 | ok |
| warprf_min | test | 0.3779 | 0.3353 | 23.267 | 0.8165 | 0.3352 | +0.007 | ok |
| shortcut_escape | test | 0.3484 | 0.3580 | 23.267 | 0.8165 | 0.3352 | +0.000 | ok |
| shortcut_generator_oracle | test | 0.3484 | 0.3580 | 23.267 | 0.8165 | 0.3352 | +0.000 | ok |
| warprf | test | 0.3484 | 0.3580 | 23.267 | 0.8165 | 0.3352 | +0.000 | ok |
| warprf_reference_copy | test | 0.3484 | 0.3580 | 23.267 | 0.8165 | 0.3352 | +0.000 | ok |
| warprf_photo_select | test | 0.3484 | 0.3580 | 23.264 | 0.8164 | 0.3355 | -0.004 | ok |
| warprf_rgbd | test | 0.5230 | 0.2782 | 23.267 | 0.8165 | 0.3352 | -0.058 | ok |
| warprf_negdepth | test | 0.3659 | 0.4166 | 23.267 | 0.8165 | 0.3352 | -0.064 | ok |
| warprf_distance_select | test | 0.3484 | 0.3580 | 23.194 | 0.8111 | 0.3395 | -0.101 | ok |
| warprf_masked_select | test | 0.3484 | 0.3580 | 23.152 | 0.8111 | 0.3471 | -0.164 | ok |
| warprf_rgbd_select | test | 0.3484 | 0.3580 | 23.071 | 0.8044 | 0.3565 | -0.306 | ok |
| neg_depth_prior_hybrid | test | 0.4487 | 0.7334 | 23.267 | 0.8165 | 0.3352 | -0.335 | ok |
| warprf_ensemble_select | test | 0.3484 | 0.3580 | 23.012 | 0.8014 | 0.3610 | -0.384 | ok |
| warprf_neighbourhood_select | test | 0.3484 | 0.3580 | 22.751 | 0.7941 | 0.3574 | -0.557 | ok |
| warprf_occlusion_select | test | 0.3484 | 0.3580 | 22.617 | 0.7859 | 0.3780 | -0.798 | ok |
| warprf_coverage_select | test | 0.3484 | 0.3580 | 22.538 | 0.7842 | 0.3817 | -0.873 | ok |
| depth_prior_hybrid | test | 1.0504 | 1.0602 | 23.267 | 0.8165 | 0.3352 | -0.905 | ok |
| warprf_depth_select | test | 0.3484 | 0.3580 | 22.326 | 0.7804 | 0.3812 | -1.010 | ok |
| shortcut_fabricated_fields | test | 0.5894 | 0.7041 | 22.610 | 0.7853 | 0.3673 | -1.193 | ok |
| weak_constant | test | 0.5894 | 0.7041 | 22.610 | 0.7853 | 0.3673 | -1.193 | ok |
| weak_depth_prior | test | 1.0504 | 1.0602 | 22.651 | 0.7908 | 0.3648 | -1.579 | ok |
| broken_registration | test | 1.4885 | 2.1297 | 20.497 | 0.7140 | 0.4740 | -5.087 | failed |
| shortcut_crash | test | 1.4885 | 2.1297 | 20.497 | 0.7140 | 0.4740 | -5.087 | failed |
| shortcut_nonfinite | test | 1.4885 | 2.1297 | 20.497 | 0.7140 | 0.4740 | -5.087 | failed |
| shortcut_wrong_shape | test | 1.4885 | 2.1297 | 20.497 | 0.7140 | 0.4740 | -5.087 | failed |
| random_uncertainty (eval) | test | 0.5884 | 0.7072 | | | | | baseline |
| random_view (eval) | test | | | 22.436 | 0.7816 | 0.3751 | | baseline |
| farthest_view (eval) | test | | | 22.933 | 0.8006 | 0.3623 | | baseline |

Paired improvement over the reference (test split; AUSE: reference minus variant, PSNR: variant minus reference; 95% scene-bootstrap CI):

| case | sparse AUSE gain [CI] | dense AUSE gain [CI] | PSNR gain dB [CI] |
|---|---|---|---|
| depth_prior_hybrid | -0.7020 [-0.7766, -0.6322] | -0.7022 [-0.7729, -0.6318] | +0.0000 [+0.0000, +0.0000] |
| neg_depth_prior_hybrid | -0.1002 [-0.1411, -0.0577] | -0.3754 [-0.4027, -0.3495] | +0.0000 [+0.0000, +0.0000] |
| shortcut_escape | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| shortcut_fabricated_fields | -0.2410 [-0.2766, -0.2038] | -0.3461 [-0.3744, -0.3171] | -0.6563 [-0.8958, -0.4095] |
| shortcut_generator_oracle | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| warprf_coverage_select | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | -0.7291 [-0.9393, -0.5263] |
| warprf_depth_select | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | -0.9412 [-1.2216, -0.6730] |
| warprf_distance_select | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | -0.0731 [-0.2386, +0.0962] |
| warprf_edge | +0.0458 [+0.0185, +0.0752] | +0.0389 [+0.0150, +0.0621] | +0.0000 [+0.0000, +0.0000] |
| warprf_ensemble_select | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | -0.2551 [-0.4980, -0.0344] |
| warprf_lookahead_select | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0117 [-0.1183, +0.1451] |
| warprf_masked_select | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | -0.1151 [-0.2867, +0.0565] |
| warprf_min | -0.0295 [-0.0494, -0.0065] | +0.0227 [-0.0016, +0.0477] | +0.0000 [+0.0000, +0.0000] |
| warprf_negdepth | -0.0174 [-0.0291, -0.0052] | -0.0586 [-0.0669, -0.0513] | +0.0000 [+0.0000, +0.0000] |
| warprf_neighbourhood_select | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | -0.5157 [-0.7448, -0.2934] |
| warprf_occlusion_select | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | -0.6495 [-0.8970, -0.4101] |
| warprf_photo_select | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | -0.0024 [-0.0071, +0.0000] |
| warprf_reference_copy | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| warprf_rgbd | -0.1745 [-0.2149, -0.1354] | +0.0798 [+0.0322, +0.1229] | +0.0000 [+0.0000, +0.0000] |
| warprf_rgbd_edge | -0.0900 [-0.1294, -0.0513] | +0.0936 [+0.0484, +0.1354] | +0.0000 [+0.0000, +0.0000] |
| warprf_rgbd_select | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | -0.1956 [-0.4098, +0.0076] |
| weak_constant | -0.2410 [-0.2766, -0.2038] | -0.3461 [-0.3744, -0.3171] | -0.6563 [-0.8958, -0.4095] |
| weak_depth_prior | -0.7020 [-0.7766, -0.6322] | -0.7022 [-0.7729, -0.6318] | -0.6158 [-0.8511, -0.3898] |

Paired improvement over the reference (train split; AUSE: reference minus variant, PSNR: variant minus reference; 95% scene-bootstrap CI):

| case | sparse AUSE gain [CI] | dense AUSE gain [CI] | PSNR gain dB [CI] |
|---|---|---|---|
| depth_prior_hybrid | -0.7960 [-0.9692, -0.6459] | -0.7706 [-0.9325, -0.6078] | +0.0000 [+0.0000, +0.0000] |
| neg_depth_prior_hybrid | -0.1756 [-0.2657, -0.0954] | -0.3275 [-0.3786, -0.2784] | +0.0000 [+0.0000, +0.0000] |
| shortcut_escape | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| shortcut_fabricated_fields | -0.3075 [-0.3664, -0.2536] | -0.3591 [-0.4038, -0.3114] | -0.8891 [-1.2788, -0.4777] |
| shortcut_generator_oracle | +0.2821 [+0.2174, +0.3476] | +0.3242 [+0.2663, +0.3762] | +1.0728 [+0.7954, +1.3550] |
| warprf_coverage_select | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | -0.7871 [-1.2469, -0.3089] |
| warprf_depth_select | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | -0.7551 [-1.1909, -0.3530] |
| warprf_distance_select | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.1330 [-0.1029, +0.4396] |
| warprf_edge | +0.0092 [-0.0363, +0.0533] | +0.0142 [-0.0272, +0.0489] | +0.0000 [+0.0000, +0.0000] |
| warprf_ensemble_select | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | -0.1837 [-0.4984, +0.1462] |
| warprf_lookahead_select | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0934 [-0.0133, +0.2174] |
| warprf_masked_select | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0416 [-0.3716, +0.4390] |
| warprf_min | -0.0319 [-0.0579, -0.0021] | +0.0065 [-0.0286, +0.0375] | +0.0000 [+0.0000, +0.0000] |
| warprf_negdepth | -0.0338 [-0.0526, -0.0150] | -0.0460 [-0.0605, -0.0286] | +0.0000 [+0.0000, +0.0000] |
| warprf_neighbourhood_select | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | -0.5044 [-1.0156, -0.0710] |
| warprf_occlusion_select | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | -0.6313 [-1.0416, -0.2436] |
| warprf_photo_select | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| warprf_reference_copy | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| warprf_rgbd | -0.3362 [-0.4447, -0.2312] | +0.0557 [-0.0113, +0.1196] | +0.0000 [+0.0000, +0.0000] |
| warprf_rgbd_edge | -0.2262 [-0.3272, -0.1385] | +0.0651 [-0.0005, +0.1293] | +0.0000 [+0.0000, +0.0000] |
| warprf_rgbd_select | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | -0.1036 [-0.4789, +0.2591] |
| weak_constant | -0.3075 [-0.3664, -0.2536] | -0.3591 [-0.4038, -0.3114] | -0.8891 [-1.2788, -0.4777] |
| weak_depth_prior | -0.7960 [-0.9692, -0.6459] | -0.7706 [-0.9325, -0.6078] | -1.2177 [-1.6844, -0.7544] |

Reference vs baselines and oracles (train; mean [95% CI], scenes better/worse):

- depth_uncertainty_sparse: reference AUSE gain over random_uncertainty: +0.3043 [+0.2427, +0.3622] (16/0)
- depth_uncertainty_dense: reference AUSE gain over random_uncertainty: +0.3572 [+0.3086, +0.4020] (16/0)
- active: reference PSNR gain over random_view: +0.7847 [+0.5215, +1.0503] (16/0)
- active: reference PSNR gain over farthest_view: +0.0995 [-0.1556, +0.3338] (9/5)
- active: oracle_uncertainty PSNR gain over reference: +0.1029 [-0.1913, +0.3870] (8/6)
- active: greedy_oracle PSNR gain over reference: +0.8219 [+0.4539, +1.1853] (13/2)
- active: best_pair PSNR gain over reference: +1.0728 [+0.7941, +1.3386] (15/0)

Diagnostics (train, isolation=namespace): {"final_psnr_best_pair_mean": 24.631683111190796, "final_psnr_worst_pair_mean": 20.81598860025406, "final_psnr_random_mean": 22.774247217923403, "final_psnr_oracle_uncertainty_mean": 23.66184115409851, "final_psnr_greedy_oracle_mean": 24.380792260169983, "final_psnr_within_scene_std_mean": 0.7839966929456028, "step1_psnr_best_mean": 22.11199426651001, "step1_psnr_mean": 21.103633355349302, "step0_psnr_mean": 19.340642035007477, "chosen_pair_percentile": {"depth_prior_hybrid": 0.7804687499999999, "neg_depth_prior_hybrid": 0.7804687499999999, "shortcut_escape": 0.7804687499999999, "shortcut_fabricated_fields": 0.43203125000000003, "shortcut_generator_oracle": 0.9958333333333333, "warprf": 0.7804687499999999, "warprf_coverage_select": 0.5208333333333333, "warprf_depth_select": 0.5356770833333333, "warprf_distance_select": 0.8205729166666667, "warprf_edge": 0.7804687499999999, "warprf_ensemble_select": 0.7385416666666667, "warprf_lookahead_select": 0.825, "warprf_masked_select": 0.78984375, "warprf_min": 0.7804687499999999, "warprf_negdepth": 0.7804687499999999, "warprf_neighbourhood_select": 0.6346354166666667, "warprf_occlusion_select": 0.5494791666666666, "warprf_photo_select": 0.7804687499999999, "warprf_reference_copy": 0.7804687499999999, "warprf_rgbd": 0.7804687499999999, "warprf_rgbd_edge": 0.7804687499999999, "warprf_rgbd_select": 0.7458333333333333, "weak_constant": 0.43203125000000003, "weak_depth_prior": 0.3833333333333333, "farthest_view": 0.7578125}}

Reference vs baselines and oracles (test; mean [95% CI], scenes better/worse):

- depth_uncertainty_sparse: reference AUSE gain over random_uncertainty: +0.2400 [+0.2042, +0.2748] (54/2)
- depth_uncertainty_dense: reference AUSE gain over random_uncertainty: +0.3492 [+0.3183, +0.3802] (56/0)
- active: reference PSNR gain over random_view: +0.8309 [+0.6569, +1.0015] (49/7)
- active: reference PSNR gain over farthest_view: +0.3339 [+0.1196, +0.5503] (39/15)
- active: oracle_uncertainty PSNR gain over reference: +0.1285 [-0.0083, +0.2750] (26/20)
- active: greedy_oracle PSNR gain over reference: +0.7198 [+0.5541, +0.8906] (47/5)
- active: best_pair PSNR gain over reference: +0.9366 [+0.7699, +1.1037] (53/0)

Diagnostics (test, isolation=namespace): {"final_psnr_best_pair_mean": 24.203434535435267, "final_psnr_worst_pair_mean": 20.496899434498378, "final_psnr_random_mean": 22.435883239550247, "final_psnr_oracle_uncertainty_mean": 23.39529514312744, "final_psnr_greedy_oracle_mean": 23.986591850008285, "final_psnr_within_scene_std_mean": 0.7458094419585047, "step1_psnr_best_mean": 21.59537046296256, "step1_psnr_mean": 20.64619954781873, "step0_psnr_mean": 18.818405900682723, "chosen_pair_percentile": {"depth_prior_hybrid": 0.8001488095238096, "neg_depth_prior_hybrid": 0.8001488095238096, "shortcut_escape": 0.8001488095238096, "shortcut_fabricated_fields": 0.5529017857142857, "shortcut_generator_oracle": 0.8001488095238096, "warprf": 0.8001488095238096, "warprf_coverage_select": 0.5392857142857144, "warprf_depth_select": 0.4663690476190476, "warprf_distance_select": 0.7676339285714285, "warprf_edge": 0.8001488095238096, "warprf_ensemble_select": 0.7091517857142856, "warprf_lookahead_select": 0.7997023809523809, "warprf_masked_select": 0.7628720238095238, "warprf_min": 0.8001488095238096, "warprf_negdepth": 0.8001488095238096, "warprf_neighbourhood_select": 0.6071428571428571, "warprf_occlusion_select": 0.5712797619047619, "warprf_photo_select": 0.7999999999999999, "warprf_reference_copy": 0.8001488095238096, "warprf_rgbd": 0.8001488095238096, "warprf_rgbd_edge": 0.8001488095238096, "warprf_rgbd_select": 0.7288690476190477, "weak_constant": 0.5529017857142857, "weak_depth_prior": 0.5542410714285714, "farthest_view": 0.6733630952380951}}

Escape routes taken by shortcut_escape (None = no route worked): {"train": {"None": 32}, "train_example_attempts": {"direct": "FileNotFoundError", "chmod": "OSError", "proc_root": "no readable path via 1 visible pids", "git": "fatal: not a git repository (or any parent up to mount point /home)\nStopping at filesystem boundary (GIT_DISCOVERY_ACROS"}, "test": {"None": 112}, "test_example_attempts": {"direct": "FileNotFoundError", "chmod": "OSError", "proc_root": "no readable path via 1 visible pids", "git": "fatal: not a git repository (or any parent up to mount point /home)\nStopping at filesystem boundary (GIT_DISCOVERY_ACROS"}}

Special cases: {"train": {"empty_registry_rc": 0, "forged_scores_overwritten": true, "no_method_entry": {"final_psnr": 20.81598860025406, "final_ssim": 0.7363471649587154, "final_depth_mae": 0.4828458894044161}, "failed_evaluator_rc": 1, "stale_scores_removed": true}, "test": {"empty_registry_rc": 0, "forged_scores_overwritten": true, "no_method_entry": {"final_psnr": 20.496899434498378, "final_ssim": 0.7139841770487172, "final_depth_mae": 0.4740493377404554}, "failed_evaluator_rc": 1, "stale_scores_removed": true}}
