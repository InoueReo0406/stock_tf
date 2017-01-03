# -*- coding: utf-8 -*-
'''
Code based on:
https://github.com/corrieelston/datalab/blob/master/FinancialTimeSeriesTensorFlow.ipynb
'''
from __future__ import print_function

import datetime
#import urllib2
#import urllib3
import math
import os
import shutil
import operator as op
from collections import namedtuple
import numpy as np
import pandas as pd
import tensorflow as tf
import argparse
from download import Download, YahooCom, YahooJp
from brands import brand_name
from layer_log import LayerLog
from brands import nikkei225_s
from feed_cache import FeedCache


#TEST_COUNT = 250		# テスト日数, 1年間あたり，245日あると仮定
TEST_COUNT = 250		# テスト日数, 1年間あたり，245日あると仮定
TRAIN_MIN = 500		# 学習データの最低日数
TRAIN_MAX = None		# 学習データの最大日数
DAYS_BACK = 3		   # 過去何日分を計算に使用するか
#STEPS = 10000		   # 学習回数
STEPS = 10000		   # 学習回数
CHECKIN_INTERVAL = 100  # 学習の途中結果を表示する間隔
REMOVE_NIL_DATE = True  # 計算対象の日付に存在しないデータを削除する
PASS_DAYS = 10		  # 除外する古いデータの日数
DROP_RATE = 0.1		 # 学習時のドロップアウトの比率
UP_RATE = 0.07		  # 上位何パーセントを買いと判断するか
STDDEV = 1e-4		   # 学習係数
REMOVE_NEWEST_DAYS = 200 * 0	# 除外する最新のデータ日数

CLASS_LABELS = ['DOWN', 'NEUTRAL', 'UP']
CLASS_DOWN = 0
CLASS_NEUTRAL = 1
CLASS_UP = 2
CLASS_COUNT = len(CLASS_LABELS)

# 学習データに使用するパラメータのラベル
PARAM_LABELS = ['Close', 'High', 'Low', 'Volume']


# 銘柄情報の入れ物
class Stock(object):
	def __init__(self, downloadClass, code, start_days_back):
		self.downloadClass = downloadClass
		self.code = code
		self.download = self.downloadClass(self.code, auto_upload=False)
		self.start_days_back = start_days_back

	@property
	def dataframe(self):
		return self.download.dataframe

Dataset = namedtuple(
	'Dataset',
	'training_predictors training_classes test_predictors test_classes')
Environ = namedtuple('Environ', 'sess model actual_classes training_step dataset feature_data keep_prob saver')


def load_exchange_dataframes(stocks, target_brand):
	'''EXCHANGESに対応するCSVファイルをPandasのDataFrameとして読み込む。

	Returns:
		{EXCHANGES[n]: pd.DataFrame()}
	'''

	# 株価を読み込む
	datas = {}

#	import pdb; pdb.set_trace()
	for (name, stock) in stocks.items():
		print(name)
		datas[name] = stock.dataframe

	# 計算対象の日付に存在しないデータを削除する

#	import pdb; pdb.set_trace()
	if REMOVE_NIL_DATE:
		target_indexes = datas[target_brand].index
		for (exchange, data) in datas.items():
			for index in data.index:
#				import pdb; pdb.set_trace()
				if not index in target_indexes:
#					import pdb; pdb.set_trace()
					datas[exchange] = datas[exchange].drop(index)
#	import pdb; pdb.set_trace()

	return datas


def load_exchange_dataframe(exchange):
	'''exchangeに対応するCSVファイルをPandasのDataFrameとして読み込む。

	Args:
		exchange: 指標名
	Returns:
		pd.DataFrame()
	'''
	return pd.read_csv('index_{}.csv'.format(exchange), index_col='Date').sort_index()


def get_using_data(dataframes, target_brand):
	'''各指標の必要なカラムをまとめて1つのDataFrameに詰める。

	Args:
		dataframes: {key: pd.DataFrame()}
	Returns:
		pd.DataFrame()
	'''
	using_data = pd.DataFrame()
	datas = [(target_brand, dataframes[target_brand])]
	datas.extend([(exchange, dataframe) for exchange, dataframe in dataframes.items() if exchange != target_brand])

	for exchange, dataframe in datas:
		using_data['{}_Open'.format(exchange)] = dataframe['Open']
		using_data['{}_Close'.format(exchange)] = dataframe['Close']
		using_data['{}_High'.format(exchange)] = dataframe['High']
		using_data['{}_Low'.format(exchange)] = dataframe['Low']
		using_data['{}_Volume'.format(exchange)] = dataframe['Volume']
	using_data = using_data.fillna(method='ffill') #Nanを昨日のデータを同じデータを代入して補完
	using_data = using_data.sort_index(ascending=True) #日付でソート，古いから新しいデータに並び替え
	return using_data


def zscore(np_array):
	'''配列の標準化を行う
	'''
	a1 = np_array.replace(0, 1.).replace(np.inf, 1.).replace(np.nan, 1.)
	a2 = a1 - a1.mean()
	a3 = a2 / a2.std()
	return a3


def get_log_return_data(stocks, using_data):
	'''各指標について、終値を1日前との比率の対数をとって正規化する。

	Args:
		using_data: pd.DataFrame()
	Returns:
		pd.DataFrame()
	'''
	log_return_data = pd.DataFrame()
	for (name, stock) in stocks.items():
		open_column = '{}_Open'.format(name)
		close_column = '{}_Close'.format(name)
		high_column = '{}_High'.format(name)
		low_column = '{}_Low'.format(name)
		volume_column = '{}_Volume'.format(name)

		# 学習データの「終値／始値」を取得
		train_close_rates = (using_data[close_column]/using_data[close_column].shift()).values[:len(using_data[close_column]) - TEST_COUNT]
		# 小さい順にソートする
		train_close_rates.sort()
		
		# 何%以上上昇した場合に購入するかの閾値を得る

		up_index = int(len(train_close_rates) * (1. - UP_RATE))
		up_rate = train_close_rates[up_index] - 1.

		# np.log(当日終値 / 前日終値) で前日からの変化率を算出
		# 前日よりも上がっていればプラス、下がっていればマイナスになる
		log_return_data['{}_Close_RATE'.format(name)] = zscore(using_data[close_column]/using_data[close_column].shift())
		# 当日高値 / 当日始値
		log_return_data['{}_High_RATE'.format(name)] = zscore(using_data[high_column]/using_data[open_column])
		# 当日安値 / 当日始値
		log_return_data['{}_Low_RATE'.format(name)] = zscore(using_data[low_column]/using_data[open_column])
		# 当日出来高 / 前日出来高
		log_return_data['{}_Volume_RATE'.format(name)] = zscore(using_data[volume_column]/using_data[volume_column].shift())

		# 答を求める
		answers = []
		# 下がる／上がると判断する変化率
		change_rate = up_rate
		for value in (using_data[close_column] / using_data[open_column]).values:
			if value < (1 - change_rate):
				# 下がる
				answers.append(CLASS_DOWN)
			elif value > (1 + change_rate):
				# 上がる
				answers.append(CLASS_UP)
			else:
				# 変化なし
				answers.append(CLASS_NEUTRAL)
		log_return_data['{}_RESULT'.format(name)] = answers

#	import pdb; pdb.set_trace()

	return log_return_data


def build_training_data(stocks, log_return_data, target_brand, max_days_back=DAYS_BACK):
	'''学習データを作る。分類クラスは、target_brandの終値が前日に比べて上ったか下がったかの2つである。
	また全指標の終値の、当日から数えてmax_days_back日前までを含めて入力データとする。

	Args:
		log_return_data: pd.DataFrame()
		target_exchange: 学習目標とする銘柄
		max_days_back: 何日前までの終値を学習データに含めるか
		# 終値 >= 始値 なら1。それ意外は0
	Returns:
		pd.DataFrame()
	'''

	# 答を詰める
	columns = ['answer_{}'.format(label) for label in CLASS_LABELS]
	for i in range(CLASS_COUNT):
		column = columns[i]
		log_return_data[column] = 0
		indices = op.eq(log_return_data['{}_RESULT'.format(target_brand)], i)
		log_return_data.ix[indices, column] = 1

	# 各指標のカラム名を追加
	for colname, _, _ in iter_exchange_days_back(stocks, target_brand, max_days_back):
		for date_type in PARAM_LABELS:
			columns.append('{}_{}'.format(colname, date_type))

	# データ数をもとめる
	max_index = len(log_return_data)

	# 学習データを作る
	training_test_data = pd.DataFrame(columns=columns)
	for i in range(max_days_back + PASS_DAYS, max_index):
		# 先頭のデータを含めるとなぜか上手くいかないので max_days_back + PASS_DAYS で少し省く
		values = {}
		# 答を入れる
		for answer_i in range(CLASS_COUNT):
			column = columns[answer_i]
			values[column] = log_return_data[column].ix[i]
		# 学習データを入れる
		for colname, exchange, days_back in iter_exchange_days_back(stocks, target_brand, max_days_back):
			for date_type in PARAM_LABELS:
				col = '{}_{}'.format(colname, date_type)
				values[col] = log_return_data['{}_{}_RATE'.format(exchange, date_type)].ix[i - days_back]
		training_test_data = training_test_data.append(values, ignore_index=True)

	# index（日付ラベル）を引き継ぐ
	training_test_data.index = log_return_data.index[max_days_back + PASS_DAYS: max_index]
	return training_test_data


def iter_exchange_days_back(stocks, target_brand, max_days_back):
	'''指標名、何日前のデータを読むか、カラム名を列挙する。
	'''
	for (exchange, stock) in stocks.items():
		end_days_back = stock.start_days_back + max_days_back
		for days_back in range(stock.start_days_back, end_days_back):
			colname = '{}_{}'.format(exchange, days_back)
			yield colname, exchange, days_back


def split_training_test_data(num_categories, training_test_data):
	'''学習データをトレーニング用とテスト用に分割する。
	'''

	# 最新のデータを除外する
	if REMOVE_NEWEST_DAYS:
		training_test_data = training_test_data[:-REMOVE_NEWEST_DAYS]

	# 学習とテストに使用するデータ数を絞る
	if TRAIN_MAX:
		training_test_data = training_test_data[:TRAIN_MAX+TEST_COUNT]

	# 先頭のいくつかより後ろが学習データ
	predictors_tf = training_test_data[training_test_data.columns[num_categories:]]
	# 先頭のいくつかが答えデータ
	classes_tf = training_test_data[training_test_data.columns[:num_categories]]

	# 学習用とテスト用のデータサイズを求める
	training_set_size = len(training_test_data) - TEST_COUNT

	return Dataset(
		training_predictors=predictors_tf[:training_set_size],
		training_classes=classes_tf[:training_set_size],
		test_predictors=predictors_tf[training_set_size:],
		test_classes=classes_tf[training_set_size:],
	)

def tf_confusion_metrics_fscore_recall_precision(model, actual_classes, session, feed_dict, env, target_brand):
	'''与えられたネットワークの正解率などを出力する。
	Args:
		model         :
		actual_classes:答えのラベル
		session       :tf.Session
		feed_dict     :
	Returns:
		Confusion_matrix:混合行列,
			i.e. インデックス(1,2)は本当はクラス1のデータを間違えてクラス2に割り当てた数になる．
		Micro_f_score   :1-d array, tp,tn,fn,fpの合計を計算後に，f scoreを導出, 各クラスのデータサイズを考慮した値
		Macro_f_score   :(1,CLASS_COUNT), それぞれのクラスのf score, 各クラスのデータサイズを考慮しない値
		Micro_recall    :同上
		Macro_recall    :同上
		Micro_precision :同上
		Macro_precision :同上
	'''
	# 保存したモデルの読み込み
	# def trainのsave_sessしたモデルを読み込む
	restore_sess(env.saver, env.sess, target_brand)

	#正解ラベルの格納
#	actuals = tf.argmax(actual_classes, 1)
	actuals = env.sess.run(tf.argmax(actual_classes, 1), feed_dict)
	#予測したラベルの格納
	expectations = env.sess.run(tf.argmax(env.model, 1), feed_dict)
	# 計算
	confusion_matrix = tf.contrib.metrics.confusion_matrix(predictions=expectations, labels=actuals, num_classes=None, dtype=tf.int32, name=None, weights=None)
	#f score, recall, precisionの計算
	from sklearn import metrics
	#マクロはそれぞれのクラスの値を計算．平均を取ると，全体のマクロになる．
	Micro_f_score   = metrics.f1_score(y_true=actuals, y_pred=expectations, labels=None, pos_label=1, average='micro', sample_weight=None)
	Macro_f_score   = metrics.f1_score(y_true=actuals, y_pred=expectations, labels=None, pos_label=1, average=None, sample_weight=None)
	Micro_recall    = metrics.recall_score(y_true=actuals, y_pred=expectations, labels=None, pos_label=1, average='micro', sample_weight=None)
	Macro_recall    = metrics.recall_score(y_true=actuals, y_pred=expectations, labels=None, pos_label=1, average=None, sample_weight=None)
	Micro_precision = metrics.precision_score(y_true=actuals, y_pred=expectations, labels=None, pos_label=1, average='micro', sample_weight=None)
	Macro_precision = metrics.precision_score(y_true=actuals, y_pred=expectations, labels=None, pos_label=1, average=None, sample_weight=None)

	return {
		'Confusion_matrix' : confusion_matrix,
		'Micro_f_score'    : Micro_f_score,
		'Macro_f_score'    : Macro_f_score,
		'Micro_recall'     : Micro_recall,
		'Macro_recall'     : Macro_recall,
		'Micro_precision'  : Micro_precision,
		'Macro_precision'  : Macro_precision,
	}

def smarter_network(stocks, dataset, layer1, layer2):
	'''隠しレイヤー入りのもうちょっと複雑な分類モデルを返す。
	'''
	sess = tf.Session()

	num_predictors = len(dataset.training_predictors.columns)
	num_classes = len(dataset.training_classes.columns)

	feature_data = tf.placeholder("float", [None, num_predictors])
	actual_classes = tf.placeholder("float", [None, num_classes])
	keep_prob = tf.placeholder(tf.float32)

	layer_counts = [layer1, layer2, CLASS_COUNT]
	weights = []
	biases = []
	model = None
	for i, count in enumerate(layer_counts):
		# 重み付けの変数定義
		if i == 0:
			weights = tf.Variable(tf.truncated_normal([num_predictors, count], stddev=STDDEV))
		else:
			weights = tf.Variable(tf.truncated_normal([layer_counts[i - 1], count], stddev=STDDEV))
		# バイアスの変数定義
		biases = tf.Variable(tf.ones([count]))

		if model == None:
			# 一番最初のレイヤー
			model = tf.nn.relu(tf.matmul(feature_data, weights) + biases)
		else:
			if (i + 1) < len(layer_counts):
				# 最後ではないレイヤー
				model = tf.nn.relu(tf.matmul(model, weights) + biases)
			else:
				# 最終レイヤーの前には dropout を入れる
				model = tf.nn.dropout(model, keep_prob)
				model = tf.nn.softmax(tf.matmul(model, weights) + biases)

	# 予測が正しいかを計算（学習に使用する）
	cost = -tf.reduce_sum(actual_classes*tf.log(model))
	training_step = tf.train.AdamOptimizer(learning_rate=STDDEV).minimize(cost)

	saver = tf.train.Saver()

	# 変数の初期化処理
#	init = tf.initialize_all_variables()
	init = tf.global_variables_initializer()

	sess.run(init)

	return Environ(
		sess=sess,
		model=model,
		actual_classes=actual_classes,
		training_step=training_step,
		dataset=dataset,
		feature_data=feature_data,
		keep_prob=keep_prob,
		saver=saver
	)


def save_sess_dir_path(target_brand):
	return os.path.join('sess_save', target_brand)


def save_sess_file_path(target_brand):
	return os.path.join(save_sess_dir_path(target_brand), 'sess.ckpt') #*.ckpt, modelの保存のための拡張子


def save_sess(saver, sess, target_brand):
	dir_path = save_sess_dir_path(target_brand)
	# 既存のファイル（保存ディレクトリ）あったら削除
	if os.path.exists(dir_path):
		shutil.rmtree(dir_path)
	os.makedirs(dir_path)
	file_path = save_sess_file_path(target_brand)
	# 保存
	saved_path = saver.save(sess, file_path)


def restore_sess(saver, sess, target_brand):
	file_path = save_sess_file_path(target_brand)
	# 読み出し
	saver.restore(sess, file_path)


def train(load_sess, env, target_prices, target_brand):
	if load_sess:
		# 学習済みのファイルを読み込む
		restore_sess(env.saver, env.sess, target_brand)
		money, trues, falses, actual_count, deal_logs, up_expectation_dates = gamble(env, target_prices)
		score = (0., money, trues, falses, actual_count, deal_logs, up_expectation_dates)
		return score
	else:
		'''学習をsteps回おこなう。
		'''

		# 予測（model）と実際の値（actual）が一致（equal）した場合の配列を取得する
		#   結果の例: [1,1,0,1,0] 1が正解
		correct_prediction = tf.equal(
			tf.argmax(env.model, 1),
			tf.argmax(env.actual_classes, 1))
		# 結果（例：[1,1,0,1,0] 1が正解）を float にキャストして
		# 全ての平均（reduce_mean）を得る
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

		max_train_accuracy = 0
		lastScore = None
		for i in range(1, 1 + STEPS):
			env.sess.run(
				env.training_step, #学習
				feed_dict=feed_dict(env, test=False, keep_prob=(1.0 - DROP_RATE)),
			)
			if i % CHECKIN_INTERVAL == 0:

				train_accuracy = env.sess.run(
					accuracy, #評価
					feed_dict=feed_dict(env, test=False),
				)
				money, trues, falses, actual_count, deal_logs, up_expectation_dates = gamble(env, target_prices)
				true_count = trues[CLASS_UP] + falses[CLASS_UP]
				true_rate = 0.
				if true_count:
					true_rate = float(trues[CLASS_UP]) / float(true_count)

				# テストデータの開始と終了の日付を取得
				test_dates = env.dataset.test_predictors.index
				test_from_date = test_dates[0]
				test_to_date = test_dates[-1]

				print(i, '{:,d}円, True Precision{:.3f}, Train Accuracy{:.3f}, Test Date{}-{}'.format(money, true_rate, train_accuracy, test_from_date, test_to_date))
				if train_accuracy < 0.5:
					break
				else:
					max_train_accuracy = train_accuracy
					# 学習済みのデータを保存
					save_sess(env.saver, env.sess, target_brand)
				lastScore = (max_train_accuracy, money, trues, falses, actual_count, deal_logs, up_expectation_dates)

#		import pdb; pdb.set_trace()
		return lastScore


# 売買シミュレーション
def gamble(env, target_prices):
	# 予想
	expectations = env.sess.run(
		tf.argmax(env.model, 1),
		feed_dict=feed_dict(env, test=True),
	)

#	import pdb; pdb.set_trace()

	# 元金
	money = 10000 * 1000
	# 売買履歴
	deal_logs = []
	# 予想が当たった数
	trues = np.zeros(CLASS_COUNT, dtype=np.int64)
	# 予想が外れた数
	falses = np.zeros(CLASS_COUNT, dtype=np.int64)
	# 実際の結果の数
	actual_count = np.zeros(CLASS_COUNT, dtype=np.int64)
	# 実際の結果
	actual_classes = getattr(env.dataset, 'test_classes')

	up_expectation_dates = []

	# 結果の集計と売買シミュレーション
	for (i, date) in enumerate(env.dataset.test_predictors.index):
		expectation = expectations[i]
		if expectation == CLASS_UP:
			up_expectation_dates.append(date)
			# 上がる予想なので買う
			price = target_prices.download.price(date)
			if price != None:
				 # 始値
				open_value = float(price[Download.COL_OPEN])
				# 終値
				close_value = float(price[Download.COL_CLOSE])
				# 購入可能な株数
				value = money / open_value
				# 購入金額
				buy_value = int(open_value * value)
				# 売却金額
				sell_value = int(close_value * value)
				# 購入
				money -= buy_value
				# 購入手数料の支払い
				money -= buy_charge(buy_value)
				# 売却
				money += sell_value
				# 売却手数料の支払い
				money -= buy_charge(sell_value)

		actual = np.argmax(actual_classes.ix[date].values)
		if expectation == actual:
			# 当たった
			trues[expectation] += 1
		else:
			# 外れた
			falses[expectation] += 1
		actual_count[actual] += 1
		deal_logs.append([date, CLASS_LABELS[expectation], CLASS_LABELS[actual], money])

	return money, trues, falses, actual_count, deal_logs, up_expectation_dates


def feed_dict(env, test=False, keep_prob=1.):
	'''学習/テストに使うデータを生成する。
	'''
	prefix = 'test' if test else 'training'
	#getattr: Pythonでオブジェクトに対し、動的にプロパティやメソッドを追加する
	predictors = getattr(env.dataset, '{}_predictors'.format(prefix)) 
	classes = getattr(env.dataset, '{}_classes'.format(prefix))
	return {
		env.feature_data: predictors.values,
		env.actual_classes: classes.values.reshape(len(classes.values), len(classes.columns)),
		env.keep_prob: keep_prob
	}


def buy_charge(yen):
	# GOMクリック証券現物手数料
	if yen <= 100000:
		return 95
	elif yen <= 200000:
		return 105
	elif yen <= 500000:
		return 260
	elif yen <= 1000000:
		return 470
	elif yen <= 1500000:
		return 570
	elif yen <= 30000000:
		return 900
	else:
		return 960


def save_deal_logs(target_brand, deal_logs):
	save_dir = 'deal_logs'
	if not os.path.exists(save_dir):
		os.mkdir(save_dir)
	with open('{}/{}.csv'.format(save_dir, target_brand), 'w') as f:
		f.write('\n'.join([','.join([str(log) for log in logs])  for logs in deal_logs]))


def save_up_expectation_dates(target_brand, up_expectation_dates):
	file_path = 'up_expectation_dates.csv'
	brand_dates = []

	# 既存のファイルがあれば読み込む
	if os.path.exists(file_path):
		with open(file_path, 'r') as f:
			brand_dates = [line.split(',') for line in f.read().split('\n')]
	brand_dates.append([target_brand] + up_expectation_dates)

	# 保存
	with open(file_path, 'w') as f:
		data = '\n'.join(','.join(brand) for brand in brand_dates)
		f.write(data)
		print('Saved: {}'.format(file_path))


#def calculate_simulation_accuracy(trues, falses, actual_count):
#	accuracy_all = np.zeros(CLASS_COUNT, dtype=np.int64)
#	for j in range(CLASS_COUNT):
#		accuracy_all[j] = 0
#	import pdb; pdb.set_trace()
#	return tets

def calculate_mean_volume_updown(target_brand):
	with open('data/YH_JP_{}.csv'.format(target_brand), 'r') as f:
		prices_all =  f.readlines()[1:] #1行目は列名が挿入されている
		prices =  prices_all[0:TEST_COUNT] #1行目は列名が挿入されている
		count = len(prices)
		count_train = len(prices_all[TEST_COUNT:])
		prices_another =  f.readlines()[0:] #1行目は列名が挿入されている
		valumes = []
		gap = []
		for (i, price) in enumerate(prices):
			price_datas = price.split(',')
			valumes.append(int(price_datas[5]))
			if price_datas[1] != '---' and price_datas[4] != '---':
				price_open = float(price_datas[1])
				price_close = float(price_datas[4])
				gap.append(abs(price_close - price_open))
			else:
				gap.append(0)
		valume_avg = sum(valumes) / count
		gap_avg = int(sum(gap) / count)
	return valume_avg, gap_avg, count_train

def main(stocks, target_brand, layer1, layer2, load_sess, result_file=None):
	# 学習データのキャッシュ
	feed_cache = FeedCache(target_brand, str(REMOVE_NEWEST_DAYS))

	# 対象の銘柄名
	target_brand_name = brand_name(target_brand)
	
	if not feed_cache.is_exist(): #パスにキャッシュが存在しない場合
		print('Make cache')
		# 株価指標データを読み込む
		all_data  = load_exchange_dataframes(stocks, target_brand)

#		import pdb; pdb.set_trace()
		# 終値を取得
		using_data = get_using_data(all_data, target_brand)

		# データを学習に使える形式に正規化
		log_return_data = get_log_return_data(stocks, using_data)

		# 答と学習データを作る
		training_test_data = build_training_data(
			stocks, log_return_data, target_brand)
		#キャッシュへの保存
		feed_cache.save(training_test_data)
	else: #パスにキャッシュが存在する場合
		print('Exist cache')
		training_test_data = feed_cache.load()

	# 学習データをトレーニング用とテスト用に分割する
	dataset = split_training_test_data(CLASS_COUNT, training_test_data)

	if len(dataset.training_predictors) < TRAIN_MIN:
		print('[{}]{}: 学習データが少なすぎるため計算を中止'.format(target_brand, target_brand_name))
		with open('results.csv', 'a') as f:
			f.write('{},{},ERROR\n'.format(target_brand, target_brand_name))

		# レイヤー検証ログに保存
		brands = nikkei225_s
		codes = [code for (code, name, _) in brands]
		#まず初期化，640行目付近で値の代入を行う．
		layerLog = LayerLog('layer_logs', '{}_{}.csv'.format(layer1, layer2), codes)
		layerLog.add(
			target_brand,
			[-1, 0, 0, 0, 0, 0, 0, 0]
		)
		return

	print('[{}]{}'.format(target_brand, target_brand_name))

	# 機械学習のネットワークを作成
	env = smarter_network(stocks, dataset, layer1, layer2)

	# 学習
	train_accuracy, money, trues, falses, actual_count, deal_logs, up_expectation_dates = train(load_sess, env, stocks[target_brand], target_brand)

	# fscore, precision, recallの計算
	result = tf_confusion_metrics_fscore_recall_precision(env.model, env.actual_classes, env.sess, feed_dict(env, True),env, target_brand)
	'''与えられたネットワークの正解率などを出力する。
	Returns:
		Confusion_matrix:混合行列,
			i.e. インデックス(1,2)は本当はクラス1のデータを間違えてクラス2に割り当てた数になる．
		Micro_f_score   :1-d array, tp,tn,fn,fpの合計を計算後に，f scoreを導出, 各クラスのデータサイズを考慮した値
		Macro_f_score   :(1,CLASS_COUNT), それぞれのクラスのf score, 各クラスのデータサイズを考慮しない値
		Micro_recall    :同上
		Macro_recall    :同上
		Micro_precision :同上
		Macro_precision :同上
	'''
	print('-- テスト --')
#	# 各クラスの適合率, 再現率, f値
#	precision = np.zeros(CLASS_COUNT, dtype=np.float64)
#	recall = np.zeros(CLASS_COUNT, dtype=np.float64)
#	macro_f_score = np.zeros(CLASS_COUNT, dtype=np.float64)
#	# 各クラスの正解数
	counts = np.zeros(CLASS_COUNT, dtype=np.int64)
	for i in range(CLASS_COUNT):
		counts[i] = trues[i] + falses[i]
#		if counts[i]:
#			# 各クラスの正解率（予想数 / 正解数）,Micro
#			precision[i] = int(float(trues[i]) / float(counts[i]) * 100)
#			recall[i] = (float(trues[i]) / float(actual_count[i])) * 100
#			macro_f_score[i] = 2*(precision[i]*recall[i])/(precision[i]+recall[i])

	#output
	print('下げ Fscore	: {:.3f} 予想{}回 正解{}回'.format(result['Macro_f_score'][CLASS_DOWN], counts[CLASS_DOWN],actual_count[CLASS_DOWN],))
	print('変化なし Fscore: {:.3f} 予想{}回 正解{}回'.format(result['Macro_f_score'][CLASS_NEUTRAL], counts[CLASS_NEUTRAL],actual_count[CLASS_NEUTRAL],))
	print('上げ Fscore	: {:.3f} 予想{}回 正解{}回'.format(result['Macro_f_score'][CLASS_UP], counts[CLASS_UP],actual_count[CLASS_UP],))
	print('上げ予測日	: {}'.format(up_expectation_dates))

	print('-- 売買シミュレーション --')
	print('売買シミュレーション結果 {:,d}円'.format(money))

	#シュミレーションの判別率の計算
#	test = calculate_simulation_accuracy(trues, falses, actual_count)
#平均取引高,平均株価上下幅(終値-始値),trainのデータ日数の計算
	valume_avg, gap_avg, count_train = calculate_mean_volume_updown(target_brand)

	# 結果をファイル保存
	#列名の入力
	if not os.path.exists(result_file):
		outputName = 'コード,名称,' \
		 + '金額,DOWN(TRUE),DOWN(FALSE),NEUTRAL(TRUE),NEUTRAL(FALSE),UP(TRUE),UP(FALSE),' \
		  + 'DOWN(PRECISION),NEUTRAL(PRECISION),UP(PRECISION),' \
		   + 'DOWN(RECALL),NEUTRAL(RECALL),UP(RECALL),' \
		    + 'DOWN(Fscore),NEUTRAL(Fscore),UP(Fscore),' \
		     + 'PRECISION(Macro),RECALL(Macro),Fscore(Macro),' \
		      + 'PRECISION(Micro),RECALL(Micro),Fscore(Micro),' \
		       + '判別率.test(日数),train(日数),平均取引高,平均株価上下幅(円)'
		with open(result_file, 'w') as f:
			f.write(outputName + '\n')

	if result_file:
		with open(result_file, 'a') as f: #a:append
			f.write('{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n'.format(
				target_brand,                             # コード
				target_brand_name,                        # 名称
				money,                                    # 金額,
				trues[CLASS_DOWN],                        # DOWN(TRUE),
				falses[CLASS_DOWN],                       # DOWN(FALSE),
				trues[CLASS_NEUTRAL],                     # NEUTRAL(TRUE),
				falses[CLASS_NEUTRAL],                    # NEUTRAL(FALSE),
				trues[CLASS_UP],                          # UP(TRUE),
				falses[CLASS_UP],                         # UP(FALSE),
				result['Macro_precision'][CLASS_DOWN],    # DOWN(PRECISION),
				result['Macro_precision'][CLASS_NEUTRAL], # NEUTRAL(PRECISION),
				result['Macro_precision'][CLASS_UP],      # UP(PRECISION),
				result['Macro_recall'][CLASS_DOWN],       # DOWN(RECALL),
				result['Macro_recall'][CLASS_NEUTRAL],    # NEUTRAL(RECALL),
				result['Macro_recall'][CLASS_UP],         # UP(RECALL),
				result['Macro_f_score'][CLASS_DOWN],      # DOWN(Fscore),
				result['Macro_f_score'][CLASS_NEUTRAL],   # NEUTRAL(Fscore),
				result['Macro_f_score'][CLASS_UP],        # UP(Fscore),
				result['Macro_precision'].mean(),         # PRECISION(Macro),
				result['Macro_recall'].mean(),            # RECALL(Macro),
				result['Macro_f_score'].mean(),           # Fscore(Macro),
				result['Micro_precision'],                # PRECISION(Micro),
				result['Micro_recall'],                   # RECALL(Micro),
				result['Micro_f_score'],                  # Fscore(Micro),
				trues.sum()/actual_count.sum(),           # 判別率,
				TEST_COUNT,                               # test(日数),
				count_train,                              # train(日数),
				valume_avg,                               # 平均取引高,
				gap_avg                                   # 平均株価上下幅(円)
				))
	else:
		print(
				target_brand,                             # コード
				target_brand_name,                        # 名称
				money,                                    # 金額,
				trues[CLASS_DOWN],                        # DOWN(TRUE),
				falses[CLASS_DOWN],                       # DOWN(FALSE),
				trues[CLASS_NEUTRAL],                     # NEUTRAL(TRUE),
				falses[CLASS_NEUTRAL],                    # NEUTRAL(FALSE),
				trues[CLASS_UP],                          # UP(TRUE),
				falses[CLASS_UP],                         # UP(FALSE),
				result['Macro_precision'][CLASS_DOWN],    # DOWN(PRECISION),
				result['Macro_precision'][CLASS_NEUTRAL], # NEUTRAL(PRECISION),
				result['Macro_precision'][CLASS_UP],      # UP(PRECISION),
				result['Macro_recall'][CLASS_DOWN],       # DOWN(RECALL),
				result['Macro_recall'][CLASS_NEUTRAL],    # NEUTRAL(RECALL),
				result['Macro_recall'][CLASS_UP],         # UP(RECALL),
				result['Macro_f_score'][CLASS_DOWN],      # DOWN(Fscore),
				result['Macro_f_score'][CLASS_NEUTRAL],   # NEUTRAL(Fscore),
				result['Macro_f_score'][CLASS_UP],        # UP(Fscore),
				result['Macro_precision'].mean(),         # PRECISION(Macro),
				result['Macro_recall'].mean(),            # RECALL(Macro),
				result['Macro_f_score'].mean(),           # Fscore(Macro),
				result['Micro_precision'],                # PRECISION(Micro),
				result['Micro_recall'],                   # RECALL(Micro),
				result['Micro_f_score'],                  # Fscore(Micro),
				TEST_COUNT,                               # test(日数),
				count_train,                              # train(日数),
				valume_avg,                               # 平均取引高,
				gap_avg                                   # 平均株価上下幅(円)
			)
	import pdb; pdb.set_trace()

	# 売買履歴をファイルに保存
	save_deal_logs(target_brand, deal_logs)

	# レイヤー検証ログに保存
	brands = nikkei225_s
	codes = [code for (code, name, _) in brands]
	layerLog = LayerLog('layer_logs', '{}_{}.csv'.format(layer1, layer2), codes)
	layerLog.add(
		target_brand,
		[
			train_accuracy,
			money,
			trues[CLASS_DOWN], falses[CLASS_DOWN],
			trues[CLASS_NEUTRAL], falses[CLASS_NEUTRAL],
			trues[CLASS_UP], falses[CLASS_UP]
		]
	)
	sys.exit()
	# 購入予想日を保存する
	save_up_expectation_dates(target_brand, up_expectation_dates)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('target_brand')
	parser.add_argument('--load_sess', type=int, default=0)
	parser.add_argument('--layer1', type=int, default=512)
	parser.add_argument('--layer2', type=int, default=512)
	args = parser.parse_args()

	stocks = {
		# 株価指標
		'DOW': Stock(YahooCom, '^DJI', 1),
		'FTSE': Stock(YahooCom, '^FTSE', 1),
		'GDAXI': Stock(YahooCom, '^GDAXI', 1),
		'HSI': Stock(YahooCom, '^HSI', 1),
		'N225': Stock(YahooCom, '^N225', 1),
		'NASDAQ': Stock(YahooCom, '^IXIC', 1),
		'SP500': Stock(YahooCom, '^GSPC', 1),
		#'SSEC': Stock(YahooCom, '000001.SS', 1),
		# 対象の銘柄
		args.target_brand: Stock(YahooJp, args.target_brand, 1)
	}
	print('REMOVE_NEWEST_DAYS {}'.format(REMOVE_NEWEST_DAYS))
	main(stocks, args.target_brand, args.layer1, args.layer2, args.load_sess, result_file='results_test.csv')
