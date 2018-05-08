$ python3 main.py 
[*] Loading training set...
[*] 12 faces loaded! 0 non-faces loaded!
[*] Loading training set successfully!
[*] Generating features...
100%|████████████████████████████████████████████████████████████████| 2/2 [00:00<00:00, 21.55it/s]
100%|████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 33.21it/s]
100%|████████████████████████████████████████████████████████████████| 2/2 [00:00<00:00, 67.52it/s]
[*] Generated 24864 features!
100%|██████████████████████████████████████████████████████████████| 12/12 [01:30<00:00,  7.56s/it]
[*] Selecting classifiers...

 11%|█████▊                                                 | 2648/24864 [10:16<1:26:13,  4.29it/s]
/Users/macbookpro/bw/face_detect/src/violajones/AdaBoost.py:117: RuntimeWarning: divide by zero encountered in log
  feature_weight = 0.5 * np.log((1 - best_error) / best_error)
 11%|█████▊                                                 | 2649/24864 [10:16<1:26:13,  4.29it/s]
/Users/macbookpro/bw/face_detect/src/violajones/AdaBoost.py:102: RuntimeWarning: divide by zero encountered in double_scalars
  weights *= 1. / np.sum(weights)
/Users/macbookpro/bw/face_detect/src/violajones/AdaBoost.py:102: RuntimeWarning: invalid value encountered in multiply
  weights *= 1. / np.sum(weights)
100%|████████████████████████████████████████████████████████| 24864/24864 [51:40<00:00,  8.02it/s]
[*] Loading test set...
[*] Loading test set successfully!
[*] 16 faces loaded! 0 non-faces loaded!
[*] Start testing...
main.py:145: RuntimeWarning: divide by zero encountered in long_scalars
  acc_neg = float(pred_neg / negative_test_img_num)
[*] Test done!
Faces 8/16 accuracy: 0.5
Objects 8/0 accuracy: inf