# -*- coding: utf-8 -*-
import os
from brands import nikkei225_excellent
from brands import (
	nikkei225_excellent5, nikkei225_excellent10,
	nikkei225_excellent20, nikkei225_excellent30
)
import argparse

#上位の銘柄だけ計算
brands = nikkei225_excellent5
#brands = nikkei225_excellent10
#brands = nikkei225_excellent20
#brands = nikkei225_excellent30
#layer1 = 512 #隠れ層
#layer2 = 512 #隠れ層
layer1 = 50 #隠れ層
layer2 = 50 #隠れ層

"""20161228 編集
全てのデータの判別率を調べる
"""
#from brands import all_brands
#brands = all_brands


file_path = 'up_expectation_dates.csv'
if os.path.exists(file_path):
	os.remove(file_path)

parser = argparse.ArgumentParser()
parser.add_argument('--skip', type=int, default=0)
args = parser.parse_args()

for i, (code, name, _) in enumerate(brands[args.skip:]):
	print('{} / {}: {} {}'.format(i + 1, len(brands), code, name))
	commena = 'py goognet.py {} --layer1={} --layer2={} --load_sess=0'.format(code, layer1, layer2)
	print(commena)
	os.system(commena) #コマンドの実行
