""" 
FLOPS = flops / sec
Ref.
https://github.com/jonggyujang0123/Early_exit/blob/master/Net_Simulator/main.ipynb
https://en.wikipedia.org/wiki/Apple_M1
https://www.itcreations.com/nvidia-gpu/nvidia-geforce-rtx-2080-super-gpu
https://www.techpowerup.com/gpu-specs/geforce-rtx-3090.c3622
https://gadgetversus.com/smartphone/samsung-galaxy-s23-ultra-specs/
https://appleinsider.com/articles/22/09/26/how-iphone-speeds-have-grown-in-the-last-5-years
https://elinux.org/Raspberry_Pi_VideoCore_APIs
https://www.gsmarena.com/here_are_the_battery_capacities_of_the_new_iphone_14_models-news-55746.php
https://en.wikipedia.org/wiki/Huawei_Mate_60
https://www.gsmarena.com/huawei_mate_60_pro-12530.php
    voltage = 3.865  # V, calculated using iPhone 14 battery capacity in mAh and Wh.
"""

from .base import *


def get_config(name):
    func = globals()[name]
    return func
