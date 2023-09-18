# 230918
# 3rd accred ## 5. ReID 정확도

# 5-1. model run
python test_clipreid.py --config_file configs/person/vit_clipreid_KETI.yml MODEL.DIST_TRAIN False MODEL.DEVICE_ID "('0')"

# 5-2. image sorting run
# notebook ./utils/error_get.ipynb
# notebook ./utils/positive_get.ipynb

# out dir
# syh '/media/syh/hdd/Clip_ReID_KETI/positive_case_result_KETI'
# syh '/media/syh/hdd/Clip_ReID_KETI/error_case_result_KETI'
#