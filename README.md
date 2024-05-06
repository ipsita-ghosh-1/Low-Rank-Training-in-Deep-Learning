# SVD_Pruning
This is a repertory for source code of Low Rank Training for Deep Neural Networks

The following are the commands to run the algorithm step-wise.

# Start up code

## train from scratch

### resnet-20

* channel

  ```
  CUDA_VISIBLE_DEVICES=2 nohup python3 cifar10_resnet20.py --save_path ./pretrained_model --decay 0 --perp_weight 1.0 --sensitivity 1e-12 --reg_type None --train --n_svd_s1 1>pretrain.out 2>&1 &
  ```

* spatial

  ```
  CUDA_VISIBLE_DEVICES=2 nohup python3 cifar10_resnet20.py --save_path ./pretrained_model_sp --decay 0 --perp_weight 1.0 --sensitivity 1e-12 --reg_type None --train --n_svd_s1 --dectype space 1>pretrain_sp.out 2>&1 &
  ```


## train with pretraining

### resnet-20

#### channel

* Hoyer

  ```
  CUDA_VISIBLE_DEVICES=1 nohup python3 cifar10_resnet20.py --save_path ./Hoyer_Model/007 --load_path ./pretrained_model/SVD_Model.pth --decay 0.07 --perp_weight 1.0 -e 1e-4 --reg_type Hoyer --train --n_svd_s1 1>hoyer_007.out 2>&1 &
  ```

* Hoyer-Square

  ```
  CUDA_VISIBLE_DEVICES=3 nohup python3 cifar10_resnet20.py --save_path ./Hoyer-Square_Model --load_path ./pretrained_model/SVD_Model.pth --decay 1e-2 --perp_weight 1.0 --sensitivity 1e-1 --reg_type Hoyer-Square --train --n_svd_s1 1>hoyer_square.out 2>&1 &
  ```

* L1

  ```
  CUDA_VISIBLE_DEVICES=0 nohup python3 cifar10_resnet20.py --save_path ./L1_Model/03 --load_path ./pretrained_model/SVD_Model.pth --decay 3e-1 --perp_weight 1.0 -e 1e-4 --reg_type L1 --train --n_svd_s1 1>L1_03.out 2>&1 &
  ```



#### spatial

- Hoyer

  ```
  CUDA_VISIBLE_DEVICES=0 nohup python3 cifar10_resnet20.py --save_path ./Hoyer_Model_sp/01 --load_path ./pretrained_model_sp/SVD_Model.pth --decay 0.1 --perp_weight 1.0 -e 4e-6 --reg_type Hoyer --train --n_svd_s1 --dectype space 1>hoyer_sp_01.out 2>&1 &
  ```

- Hoyer-Square

  ```
  CUDA_VISIBLE_DEVICES=3 nohup python3 cifar10_resnet20.py --save_path ./Hoyer-Square_Model_sp --load_path ./pretrained_model_sp/SVD_Model.pth --decay 1e-2 --perp_weight 1.0 --sensitivity 3e-2 --reg_type Hoyer-Square --train --n_svd_s1 --dectype space 1>hoyer_square_sp.out 2>&1 &
  ```

- L1

  ```
  CUDA_VISIBLE_DEVICES=3 nohup python3 cifar10_resnet20.py --save_path ./L1_Model_sp/03 --load_path ./pretrained_model_sp/SVD_Model.pth --decay 0.3 --perp_weight 1.0 -e 1e-4 --reg_type L1 --train --n_svd_s1 --dectype space 1>L1_sp_03.out 2>&1 &
  ```


## test and prune

### resnet-20

#### channel

* None

  ```
  python3 cifar10_resnet20.py --load_path ./pretrained_model/SVD_Model.pth --save_path ./pretrained_model -e 3e-1 --test --n_svd_s1
  ```

* Hoyer

  ```
  CUDA_VISIBLE_DEVICES=5 python3 cifar10_resnet20.py --load_path ./Hoyer_Model/01/SVD_Model.pth --save_path ./Hoyer_Model/01 -e 3e-6 --test --n_svd_s1
  ```

* Hoyer-Square

  ```
  CUDA_VISIBLE_DEVICES=2 python3 cifar10_resnet20.py --load_path ./Hoyer-Square_Model/0007/SVD_Model.pth --save_path ./Hoyer-Square_Model/0007 -e 5e-6 --test --n_svd_s1
  ```

* L1

  ```
  CUDA_VISIBLE_DEVICES=2 python3 cifar10_resnet20.py --load_path ./L1_Model/03/SVD_Model.pth --save_path ./L1_Model/03 -e 1e-1 --test --n_svd_s1
  ```



#### spatial

* None

  ```
  python3 cifar10_resnet20.py --load_path ./pretrained_model_sp/SVD_Model.pth --save_path ./pretrained_model_sp -e 3e-1 --test --n_svd_s1 --dectype space
  ```

* Hoyer

  ```
  python3 cifar10_resnet20.py --load_path ./Hoyer_Model_sp/001/SVD_Model.pth --save_path ./Hoyer_Model_sp/001 -e 1e-1 --test --n_svd_s1 --dectype space
  ```

* Hoyer-Square

  ```
  python3 cifar10_resnet20.py --load_path ./Hoyer-Square_Model_sp/SVD_Model.pth --save_path ./Hoyer-Square_Model_sp --sensitivity 0.1 --test --n_svd_s1 --dectype space
  ```

* L1

  ```
  python3 cifar10_resnet20.py --load_path ./L1_Model_sp/03/SVD_Model.pth --save_path ./L1_Model_sp/03 -e 1e-1 --test --n_svd_s1 --dectype space
  ```

## fine tune

### resnet 20

#### channel

* Hoyer

  ```
  CUDA_VISIBLE_DEVICES=2 nohup python3 cifar10_resnet20.py --save_path ./Hoyer_FT_Model/01 --load_path ./Hoyer_Model/01/SVD_pruning_Model.pth --decay 0.0 --perp_weight 1.0 --reg_type None --train --n_svd_s1 --prun 1>hoyerFT_01.out 2>&1 &
  ```

* Hoyer-Square

  ```
  CUDA_VISIBLE_DEVICES=2 nohup python3 cifar10_resnet20.py --save_path ./Hoyer-Square_FT_Model --load_path ./Hoyer-Square_Model/SVD_pruning_Model.pth --decay 0.0 --perp_weight 1.0 --reg_type None --train --n_svd_s1 --prun 1>hoyerSFT.out 2>&1 &
  ```

* L1

  ```
  CUDA_VISIBLE_DEVICES=2 nohup python3 cifar10_resnet20.py --save_path ./L1_FT_Model/03 --load_path ./L1_Model/03/SVD_pruning_Model.pth --decay 0.0 --perp_weight 1.0 --reg_type None --train --n_svd_s1 --prun 1>L1FT_03.out 2>&1 &
  ```



#### spatial

* None

  ```
  CUDA_VISIBLE_DEVICES=0 nohup python3 cifar10_resnet20.py --save_path ./None_FT_Model_sp --load_path ./pretrained_model_sp/SVD_pruning_Model.pth --decay 0.0 --perp_weight 1.0 --reg_type None --train --n_svd_s1 --prun --dectype space 1>NoneFT_sp.out 2>&1 &
  ```

* Hoyer

  ```
  CUDA_VISIBLE_DEVICES=0 nohup python3 cifar10_resnet20.py --save_path ./Hoyer_FT_Model_sp/001 --load_path ./Hoyer_Model_sp/001/SVD_pruning_Model.pth --decay 0.0 --perp_weight 1.0 --reg_type None --train --n_svd_s1 --prun --dectype space 1>hoyerFT_sp.out_001 2>&1 &
  ```

* Hoyer-Square

  ```
  CUDA_VISIBLE_DEVICES=2 nohup python3 cifar10_resnet20.py --save_path ./Hoyer-Square_FT_Model_sp --load_path ./Hoyer-Square_Model_sp/SVD_pruning_Model.pth --decay 0.0 --perp_weight 1.0 --reg_type None --train --n_svd_s1 --prun --dectype space 1>hoyerSFT_sp.out 2>&1 &
  ```

* L1

  ```
  CUDA_VISIBLE_DEVICES=2 nohup python3 cifar10_resnet20.py --save_path ./L1_FT_Model_sp/03 --load_path ./L1_Model_sp/03/SVD_pruning_Model.pth --decay 0.0 --perp_weight 1.0 --reg_type None --train --n_svd_s1 --prun --dectype space 1>L1FT_sp_03.out 2>&1 &
  ```
