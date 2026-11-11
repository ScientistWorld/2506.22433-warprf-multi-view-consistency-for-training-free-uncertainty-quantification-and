| case | split | sparse AUSE | dense AUSE | final PSNR | final SSIM | final depth MAE | cohort rank | status |
|---|---|---:|---:|---:|---:|---:|---:|---|
| shortcut_generator_oracle | train | 0.0000 | 0.0000 | 24.283 | 0.8417 | 0.2297 | +1.965 | ok |
| shortcut_read_hidden_answers | train | 0.0000 | 0.0000 | 24.283 | 0.8417 | 0.2297 | +1.965 | ok |
| warprf_edge | train | 0.3831 | 0.3857 | 22.946 | 0.8173 | 0.2374 | +0.101 | ok |
| depth_prior_hybrid | train | 0.3223 | 0.4587 | 22.946 | 0.8173 | 0.2374 | +0.092 | ok |
| warprf_rgbd_edge | train | 0.4937 | 0.3425 | 22.946 | 0.8173 | 0.2374 | +0.051 | ok |
| warprf_min | train | 0.4646 | 0.4039 | 22.946 | 0.8173 | 0.2374 | +0.029 | ok |
| warprf_rgbd | train | 0.5241 | 0.3476 | 22.946 | 0.8173 | 0.2374 | +0.020 | ok |
| warprf | train | 0.4301 | 0.4489 | 22.946 | 0.8173 | 0.2374 | +0.000 | ok |
| warprf_reference_copy | train | 0.4301 | 0.4489 | 22.946 | 0.8173 | 0.2374 | +0.000 | ok |
| warprf_ensemble_select | train | 0.4301 | 0.4489 | 22.911 | 0.8128 | 0.2400 | -0.085 | ok |
| warprf_rgbd_select | train | 0.4301 | 0.4489 | 22.835 | 0.8129 | 0.2415 | -0.141 | ok |
| warprf_plus | train | 0.4937 | 0.3425 | 22.763 | 0.8119 | 0.2429 | -0.153 | ok |
| warprf_lookahead_select | train | 0.4301 | 0.4489 | 22.763 | 0.8119 | 0.2429 | -0.204 | ok |
| warprf_masked | train | 0.4337 | 0.4504 | 22.935 | 0.8074 | 0.2492 | -0.206 | ok |
| weak_depth_prior | train | 0.3223 | 0.4587 | 22.703 | 0.8002 | 0.2530 | -0.345 | ok |
| warprf_coverage_select | train | 0.4301 | 0.4489 | 22.485 | 0.7959 | 0.2590 | -0.656 | ok |
| warprf_depth_select | train | 0.4301 | 0.4489 | 22.518 | 0.7914 | 0.2625 | -0.710 | ok |
| shortcut_fabricated_fields | train | 0.5639 | 0.6059 | 22.426 | 0.7886 | 0.2633 | -1.030 | ok |
| weak_constant | train | 0.5639 | 0.6059 | 22.426 | 0.7886 | 0.2633 | -1.030 | ok |
| broken_registration | train | 1.3354 | 1.6308 | 20.963 | 0.7450 | 0.3087 | -4.180 | failed |
| shortcut_crash | train | 1.3354 | 1.6308 | 20.963 | 0.7450 | 0.3087 | -4.180 | failed |
| shortcut_nonfinite | train | 1.3354 | 1.6308 | 20.963 | 0.7450 | 0.3087 | -4.180 | failed |
| shortcut_wrong_shape | train | 1.3354 | 1.6308 | 20.963 | 0.7450 | 0.3087 | -4.180 | failed |
| random_uncertainty (eval) | train | 0.5621 | 0.6006 | | | | | baseline |
| random_view (eval) | train | | | 22.533 | 0.7974 | 0.2537 | | baseline |
| farthest_view (eval) | train | | | 22.904 | 0.8085 | 0.2469 | | baseline |
| shortcut_read_hidden_answers | test | 0.0000 | 0.0000 | 24.177 | 0.8331 | 0.2160 | +2.211 | ok |
| warprf_rgbd_edge | test | 0.4889 | 0.3386 | 23.012 | 0.8035 | 0.2327 | +0.097 | ok |
| warprf_edge | test | 0.4243 | 0.4017 | 23.012 | 0.8035 | 0.2327 | +0.094 | ok |
| warprf_plus | test | 0.4889 | 0.3386 | 22.970 | 0.8039 | 0.2311 | +0.093 | ok |
| depth_prior_hybrid | test | 0.3936 | 0.4546 | 23.012 | 0.8035 | 0.2327 | +0.080 | ok |
| warprf_rgbd | test | 0.5149 | 0.3443 | 23.012 | 0.8035 | 0.2327 | +0.068 | ok |
| warprf_masked | test | 0.4699 | 0.4626 | 23.114 | 0.8031 | 0.2318 | +0.067 | ok |
| warprf_min | test | 0.4631 | 0.4169 | 23.012 | 0.8035 | 0.2327 | +0.059 | ok |
| shortcut_generator_oracle | test | 0.4633 | 0.4618 | 23.012 | 0.8035 | 0.2327 | +0.000 | ok |
| warprf | test | 0.4633 | 0.4618 | 23.012 | 0.8035 | 0.2327 | +0.000 | ok |
| warprf_reference_copy | test | 0.4633 | 0.4618 | 23.012 | 0.8035 | 0.2327 | +0.000 | ok |
| warprf_lookahead_select | test | 0.4633 | 0.4618 | 22.970 | 0.8039 | 0.2311 | -0.004 | ok |
| warprf_ensemble_select | test | 0.4633 | 0.4618 | 22.891 | 0.8011 | 0.2287 | -0.054 | ok |
| warprf_rgbd_select | test | 0.4633 | 0.4618 | 22.875 | 0.8005 | 0.2347 | -0.138 | ok |
| weak_depth_prior | test | 0.3936 | 0.4546 | 22.719 | 0.7901 | 0.2371 | -0.280 | ok |
| shortcut_fabricated_fields | test | 0.5645 | 0.6299 | 22.702 | 0.7939 | 0.2329 | -0.510 | ok |
| weak_constant | test | 0.5645 | 0.6299 | 22.702 | 0.7939 | 0.2329 | -0.510 | ok |
| warprf_coverage_select | test | 0.4633 | 0.4618 | 22.526 | 0.7815 | 0.2463 | -0.667 | ok |
| warprf_depth_select | test | 0.4633 | 0.4618 | 22.226 | 0.7692 | 0.2601 | -1.126 | ok |
| broken_registration | test | 1.3394 | 1.7109 | 21.073 | 0.7344 | 0.2894 | -4.335 | failed |
| shortcut_crash | test | 1.3394 | 1.7109 | 21.073 | 0.7344 | 0.2894 | -4.335 | failed |
| shortcut_nonfinite | test | 1.3394 | 1.7109 | 21.073 | 0.7344 | 0.2894 | -4.335 | failed |
| shortcut_wrong_shape | test | 1.3394 | 1.7109 | 21.073 | 0.7344 | 0.2894 | -4.335 | failed |
| random_uncertainty (eval) | test | 0.5643 | 0.6258 | | | | | baseline |
| random_view (eval) | test | | | 22.594 | 0.7878 | 0.2399 | | baseline |
| farthest_view (eval) | test | | | 22.969 | 0.7991 | 0.2304 | | baseline |

Paired improvement over the reference (test split; AUSE: reference minus variant, PSNR: variant minus reference; 95% scene-bootstrap CI):

| case | sparse AUSE gain [CI] | dense AUSE gain [CI] | PSNR gain dB [CI] |
|---|---|---|---|
| depth_prior_hybrid | +0.0697 [+0.0213, +0.1187] | +0.0072 [-0.0558, +0.0683] | +0.0000 [+0.0000, +0.0000] |
| shortcut_fabricated_fields | -0.1013 [-0.1378, -0.0633] | -0.1680 [-0.2027, -0.1316] | -0.3097 [-0.6556, +0.0249] |
| shortcut_generator_oracle | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| shortcut_read_hidden_answers | +0.4633 [+0.4247, +0.4993] | +0.4618 [+0.4165, +0.5054] | +1.1654 [+0.9186, +1.4207] |
| warprf_coverage_select | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | -0.4860 [-0.8306, -0.1660] |
| warprf_depth_select | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | -0.7854 [-1.1478, -0.4447] |
| warprf_edge | +0.0390 [+0.0176, +0.0624] | +0.0602 [+0.0314, +0.0872] | +0.0000 [+0.0000, +0.0000] |
| warprf_ensemble_select | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | -0.1201 [-0.3715, +0.1072] |
| warprf_lookahead_select | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | -0.0411 [-0.2997, +0.1562] |
| warprf_masked | -0.0066 [-0.0151, -0.0002] | -0.0008 [-0.0015, -0.0001] | +0.1029 [-0.1119, +0.3259] |
| warprf_min | +0.0001 [-0.0262, +0.0253] | +0.0450 [+0.0060, +0.0814] | +0.0000 [+0.0000, +0.0000] |
| warprf_plus | -0.0256 [-0.0637, +0.0107] | +0.1232 [+0.0540, +0.1852] | -0.0411 [-0.2997, +0.1562] |
| warprf_reference_copy | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| warprf_rgbd | -0.0517 [-0.0935, -0.0139] | +0.1175 [+0.0495, +0.1784] | +0.0000 [+0.0000, +0.0000] |
| warprf_rgbd_edge | -0.0256 [-0.0637, +0.0107] | +0.1232 [+0.0540, +0.1852] | +0.0000 [+0.0000, +0.0000] |
| warprf_rgbd_select | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | -0.1368 [-0.3296, +0.0043] |
| weak_constant | -0.1013 [-0.1378, -0.0633] | -0.1680 [-0.2027, -0.1316] | -0.3097 [-0.6556, +0.0249] |
| weak_depth_prior | +0.0697 [+0.0213, +0.1187] | +0.0072 [-0.0558, +0.0683] | -0.2928 [-0.6926, +0.1243] |

Diagnostics (train): {"final_psnr_best_pair_mean": 24.282975554466248, "final_psnr_worst_pair_mean": 20.963368773460388, "final_psnr_random_mean": 22.533121049404144, "final_psnr_within_scene_std_mean": 0.7900605311169113, "step1_psnr_best_mean": 21.949698567390442, "step1_psnr_mean": 20.964801892638206, "step0_psnr_mean": 19.368082642555237, "chosen_pair_percentile": {"depth_prior_hybrid": 0.6473214285714286, "shortcut_fabricated_fields": 0.4720982142857143, "shortcut_generator_oracle": 0.9821428571428571, "shortcut_read_hidden_answers": 0.9821428571428571, "warprf": 0.6473214285714286, "warprf_coverage_select": 0.4575892857142857, "warprf_depth_select": 0.45982142857142855, "warprf_edge": 0.6473214285714286, "warprf_ensemble_select": 0.6573660714285714, "warprf_lookahead_select": 0.5636160714285714, "warprf_masked": 0.6383928571428571, "warprf_min": 0.6473214285714286, "warprf_plus": 0.5636160714285714, "warprf_reference_copy": 0.6473214285714286, "warprf_rgbd": 0.6473214285714286, "warprf_rgbd_edge": 0.6473214285714286, "warprf_rgbd_select": 0.6060267857142857, "weak_constant": 0.4720982142857143, "weak_depth_prior": 0.5535714285714286, "farthest_view": 0.6395089285714286}}

Diagnostics (test): {"final_psnr_best_pair_mean": 24.177013397216797, "final_psnr_worst_pair_mean": 21.073137044906616, "final_psnr_random_mean": 22.593536804829327, "final_psnr_within_scene_std_mean": 0.7183025064491402, "step1_psnr_best_mean": 21.76400715112686, "step1_psnr_mean": 20.910157836973667, "step0_psnr_mean": 19.12775808572769, "chosen_pair_percentile": {"depth_prior_hybrid": 0.6199776785714286, "shortcut_fabricated_fields": 0.50390625, "shortcut_generator_oracle": 0.6199776785714286, "shortcut_read_hidden_answers": 0.9821428571428571, "warprf": 0.6199776785714286, "warprf_coverage_select": 0.4732142857142857, "warprf_depth_select": 0.33203125, "warprf_edge": 0.6199776785714286, "warprf_ensemble_select": 0.5758928571428572, "warprf_lookahead_select": 0.62109375, "warprf_masked": 0.6713169642857143, "warprf_min": 0.6199776785714286, "warprf_plus": 0.62109375, "warprf_reference_copy": 0.6199776785714286, "warprf_rgbd": 0.6199776785714286, "warprf_rgbd_edge": 0.6199776785714286, "warprf_rgbd_select": 0.5842633928571428, "weak_constant": 0.50390625, "weak_depth_prior": 0.5474330357142858, "farthest_view": 0.6294642857142858}}

Special cases: {"train": {"empty_registry_rc": 0, "forged_scores_overwritten": true, "no_method_entry": {"final_psnr": 20.963368773460388, "final_ssim": 0.7450411915779114, "final_depth_mae": 0.308685096912086}, "failed_evaluator_rc": 1, "stale_scores_removed": true}, "test": {"empty_registry_rc": 0, "forged_scores_overwritten": true, "no_method_entry": {"final_psnr": 21.073137044906616, "final_ssim": 0.7344210725277662, "final_depth_mae": 0.2894266373477876}, "failed_evaluator_rc": 1, "stale_scores_removed": true}}
