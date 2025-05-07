
device=cuda:1
mode=POSE
config_name=progressive

python3 main.py \
--config_name $config_name \
--device $device \
--mode $mode \
--data MM_class

