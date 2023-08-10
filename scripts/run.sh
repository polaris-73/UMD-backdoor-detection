python train_contam.py --DATASET cifar10 --TC 0 --SETTING A2O --RUN 0 --DEVICE 0 --ATTACK perturbation
python est.py --DATASET cifar10 --mode pert --SETTING A2O --RUN 0 --DEVICE 0 --ATTACK perturbation
python test_trans.py --DATASET cifar10 --mode pert --SETTING A2O --RUN 0 --DEVICE 0 --ATTACK perturbation
python node_clustering.py --DATASET cifar10 --mode pert --SETTING A2O --RUN 0 --DEVICE 0 --ATTACK perturbation