# 23.11.17
#python train_promptreid.py --config_file configs/person/vit_clipreid_promptbase_msmt_server_run_dino_msmt.yml MODEL.DIST_TRAIN False
#python train_promptreid.py --config_file configs/person/vit_clipreid_promptbase_msmt_server_run_dino_imagenet.yml MODEL.DIST_TRAIN False

# 23.11.17
# Dino msmt 80epoch test
#python train_promptreid.py --config_file configs/person/vit_clipreid_promptbase_msmt_server_run_dino_msmt.yml MODEL.DIST_TRAIN False

# Dino msmt 20epoch test dino visual
#python train_promptreid.py --config_file configs/person/vit_clipreid_promptbase_msmt_server_run_dino_msmt.yml MODEL.DIST_TRAIN False

# retrain clip reid base
python train_promptreid.py --config_file configs/person/vit_clipreid_msmt.yml MODEL.DIST_TRAIN False
