# 実践的ファインチューニング

本文章の目的はファインチューニングを実際に行ってみて、実践的な手法を確立すること。
もしくはその記録。

[旧README](./README1.md)はファイルを分けた。

## 課題と、解決に向けたアプローチ

### アニメーションGIF

アニメーションGIFは最初のフレームだけを学習・推論に利用している。
そのため強いノイズになってる可能性がある。
適切なフレームを利用するか、データから除外するかしたほうが良い。

専用のツールを作った。
アニメーションGIFについては代表フレームを自動で判定し、
PaliGemmaの入力に使える224x224のサイズに変換する。
普通の画像については224x224サイズへの変換のみを行う。

<https://github.com/koron/palicnv>

このツールにより、以降の入力画像は変換済みのものとした。
学習・推論時に画像の変換は行わないので、わずかにオーバーヘッドが減った。

### データセットの偏り

データセットが偏っているのでは。
今回学習・検証に利用したデータセットは「ギリギリセーフ」と「ギリギリアウト」のデータセットになっており
「明らかにセーフ」や「明らかにアウト」なデータではない。
またこのデータセット学習後のモデルで「明らかにセーフ」や「明らかにアウト」のケースをinferenceしたら、
過学習により汎化能力を失ってることを考えると、ガタガタになる可能性が高い。
たしかめてみる。

より大きなデータセット(10388個)を用意した。
うち1615個がアウトデータであり、アウト理由が付与されている。

当初はPaliGemmaが持つLLM的な知識を活用するため、
アウト理由によるクラスタリングを試みたが断念した。
その理由は以下の通り:

* プロンプトが長くなり計算時間が延びた
* クラスタのラベルとその定義が曖昧であり、LLMの知識を活用できないと判断した
* クラスタによりサンプル数の差が激しく(一部クラスタのサンプルが極端に少ない)

結果、広告として掲載許可がでるかどうかを `yes` か `no` かで答えさせる
以下のようなプロンプトを採用した。
"Google Adsense” と具体名を入れたのは、初期学習においてGoogle Adsenseから、
なにかしらのデータが混入しているのではと期待したから。

> Answer yes or no to whether this image is approved as an ad by Google Adsense.

このプロンプトに対して、本データは 84.4% がyesと見做せる。

## 補題

### 行指向入力データをシャッフルして分割する別解

```console
shuf indata.tsv | spilt -l 1000 --additional-suffix .tsv - outdir/splitted-
```

1万行のデータを上記コマンドで処理すると、シャッフルした上で
outdir/splitted-aa.tsv から outdir/splitted-aj.tsv まで
10個のファイルに分けられる。

### 学習時のVRAM

学習のスクリプトに以下のコードを追加することで、
VRAMの割り当てを都度割り当てにし、10個程度までバッチサイズを大きくできた。

```python
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'
```

どうやらJAXはPythonのオブジェクトをVRAMに転送する際、
予めVRAMを確保したり、
そのオブジェクトが使われなくなってもキャッシュして可能な限り解放しないらしい。
結果、その後の計算に不要なメモリを持ち続け、必要なメモリを確保できないとのこと。

上記の設定はそれを必要に応じて確保するようにし、使わないVRAMは即時解放するようにしている。
結果、VRAMへのロードは増えオーバーヘッドがあるものの、必要なオブジェクトのみがVRAMに乗るため、
限られたVRAMサイズでも大きめのバッチで学習できるようになった。

## 学習の評価

分けたファイルの1つaaから先頭64個を抽出し、
バッチ10個 learning rate 0.001 で学習しパラメーターを保存。

その保存したパラメーターで分けたファイルの1つabを推定し、
学習前後の成績確認。

```console
$ ./14-validate_outtsv.py tmp/*-ab.tsv
Aggregate: tmp/out64-b10-ab.tsv
  Tp=854 Tn=17 Fp=155 Fn=13
  accuracy:  0.8383060635226179
  precision: 0.846382556987116
  recall:    0.9850057670126874
  F-measure: 0.9104477611940298
Aggregate: tmp/splitted-ab.tsv
  Tp=652 Tn=57 Fp=115 Fn=215
  accuracy:  0.6823869104908566
  precision: 0.8500651890482399
  recall:    0.7520184544405998
  F-measure: 0.7980416156670748
```

一見良くなっているように見えるがTnとFnが減りTpとFpが増えている。
これはnoだったものが学習でyesになったことを示しており
もともと正当がyesである率が高いため、
学習が成功したとは評価しづらい。
そのことはprecisionが下がっていることが裏付けている。

学習に使うデータを再検討したほうが良かろう。

aaセットから `false positive/negative` だけ、64件を取り出して学習してみる。
(答えが

```console
$ ./16-filter_train_data.py tmp/splitted-aa.tsv | head -64 > tmp/false64.tsv
$ ./15-train.py -d tmp/false64.tsv -i dataset/pub_judges -b 10 -r 0.001 -s checkpoints/false64-r0.001-b10.npz

$ ./13-inference.py -l ./checkpoints/false64-r0.001-b10.npz -d tmp/splitted-ab.tsv -i dataset/pub_judges -b 4 | tee tmp/out-false64-r0.001-b10-ab.tsv

$ ./13-inference.py -l ./checkpoints/false64-r0.001-b10.npz -d tmp/splitted-aa.tsv -i dataset/pub_judges -b 4 | tee tmp/out-false64-r0.001-b10-aa.tsv

$ ./14-validate_outtsv.py tmp/out-false64-r0.001-b10-aa.tsv tmp/out-false64-r0.001-b10-ab.tsv tmp/splitted-ab.tsv
Aggregate: tmp/out-false64-r0.001-b10-aa.tsv
  Tp=880 Tn=0 Fp=159 Fn=0
  accuracy:  0.8469682386910491
  precision: 0.8469682386910491
  recall:    1.0
  F-measure: 0.9171443460135488
Aggregate: tmp/out-false64-r0.001-b10-ab.tsv
  Tp=867 Tn=0 Fp=172 Fn=0
  accuracy:  0.8344562078922041
  precision: 0.8344562078922041
  recall:    1.0
  F-measure: 0.9097586568730325
Aggregate: tmp/splitted-ab.tsv
  Tp=652 Tn=57 Fp=115 Fn=215
  accuracy:  0.6823869104908566
  precision: 0.8500651890482399
  recall:    0.7520184544405998
  F-measure: 0.7980416156670748
```

yes とだけ返すように学習してしまったようだ。
false negative のデータが多いため (Fn=215)
yes と返すことを学習するのではと予想できる。

true positive だけを除いた64件のデータで8バッチにして8回(計64ステップ)学習してみる。

```console
$ grep -v '\bTrue\b.*\byes$' ./tmp/splitted-aa.tsv | head -64 > tmp/noTp64.tsv

$ ./15-train.py -d tmp/noTp64.tsv -i dataset/pub_judges -t 8 -b 8 -r 0.001 -s checkpoints/noTp64-t8-r0.001-b80.npz

$ ./13-inference.py -l ./checkpoints/noTp64-t8-r0.001-b8.npz -d tmp/splitted-ab.tsv -i dataset/pub_judges | tee tmp/out-noTp64-t8-r0.001-b8-ab.tsv
$ ./13-inference.py -l ./checkpoints/noTp64-t8-r0.001-b8.npz -d tmp/splitted-aa.tsv -i dataset/pub_judges | tee tmp/out-noTp64-t8-r0.001-b8-aa.tsv
$ ./13-inference.py -l ./checkpoints/noTp64-t8-r0.001-b8.npz -d tmp/noTp64.tsv      -i dataset/pub_judges | tee tmp/out-noTp64-t8-r0.001-b8-xx.tsv

$ ./14-validate_outtsv.py tmp/out-noTp64-t8-r0.001-b8-xx.tsv tmp/noTp64.tsv
Aggregate: tmp/out-noTp64-t8-r0.001-b8-xx.tsv
  Tp=41 Tn=23 Fp=0 Fn=0
  accuracy:  1.0
  precision: 1.0
  recall:    1.0
  F-measure: 1.0
Aggregate: tmp/noTp64.tsv
  Tp=0 Tn=5 Fp=18 Fn=41
  accuracy:  0.078125
  precision: 0.0
  recall:    0.0

$ ./14-validate_outtsv.py tmp/out-noTp64-t8-r0.001-b8-ab.tsv tmp/splitted-ab.tsv
Aggregate: tmp/out-noTp64-t8-r0.001-b8-ab.tsv
  Tp=813 Tn=41 Fp=131 Fn=54
  accuracy:  0.821944177093359
  precision: 0.861228813559322
  recall:    0.9377162629757786
  F-measure: 0.8978464936499171
Aggregate: tmp/splitted-ab.tsv
  Tp=652 Tn=57 Fp=115 Fn=215
  accuracy:  0.6823869104908566
  precision: 0.8500651890482399
  recall:    0.7520184544405998
  F-measure: 0.7980416156670748

$ ./14-validate_outtsv.py tmp/out-noTp64-t8-r0.001-b8-aa.tsv tmp/splitted-aa.tsv
Aggregate: tmp/out-noTp64-t8-r0.001-b8-aa.tsv
  Tp=837 Tn=47 Fp=112 Fn=43
  accuracy:  0.8508180943214629
  precision: 0.8819810326659642
  recall:    0.9511363636363637
  F-measure: 0.9152542372881356
Aggregate: tmp/splitted-aa.tsv
  Tp=648 Tn=52 Fp=107 Fn=232
  accuracy:  0.6737247353224254
  precision: 0.8582781456953642
  recall:    0.7363636363636363
  F-measure: 0.7926605504587156
```

true positive を除外しない aaからの64件のデータで、8バッチに8回(計64ステップ)学習し、傾向をみる。

```console
$ ./15-train.py -d tmp/test64.tsv -i dataset/pub_judges -b 8 -r 0.001 -t8 -s checkpoints/test64-t8-r0.001-b8.npz

$ ./13-inference.py -l checkpoints/test64-t8-r0.001-b8.npz -d tmp/splitted-ab.tsv -i dataset/pub_judges | tee tmp/out-test64-t8-r0.001-b8-ab.tsv
$ ./14-validate_outtsv.py tmp/out-test64-t8-r0.001-b8-ab.tsv ./tmp/splitted-ab.tsv
Aggregate: tmp/out-test64-t8-r0.001-b8-ab.tsv
  Tp=866 Tn=20 Fp=152 Fn=1
  accuracy:  0.8527430221366699
  precision: 0.8506876227897839
  recall:    0.9988465974625144
  F-measure: 0.9188328912466843
Aggregate: ./tmp/splitted-ab.tsv
  Tp=652 Tn=57 Fp=115 Fn=215
  accuracy:  0.6823869104908566
  precision: 0.8500651890482399
  recall:    0.7520184544405998
  F-measure: 0.7980416156670748

./13-inference.py -l checkpoints/test64-t8-r0.001-b8.npz -d tmp/splitted-aa.tsv -i dataset/pub_judges | tee tmp/out-test64-t8-r0.001-b8-aa.tsv
./14-validate_outtsv.py tmp/out-test64-t8-r0.001-b8-aa.tsv ./tmp/splitted-aa.tsv
Aggregate: tmp/out-test64-t8-r0.001-b8-aa.tsv
  Tp=878 Tn=25 Fp=134 Fn=2
  accuracy:  0.8691049085659288
  precision: 0.8675889328063241
  recall:    0.9977272727272727
  F-measure: 0.9281183932346723
Aggregate: ./tmp/splitted-aa.tsv
  Tp=648 Tn=52 Fp=107 Fn=232
  accuracy:  0.6737247353224254
  precision: 0.8582781456953642
  recall:    0.7363636363636363
  F-measure: 0.7926605504587156
```

結果がnoになるべきデータのみで学習させたら…
当然すべてにnoと返すようになった。
そりゃそうだわw

なお以降、学習は8バッチの8イテレーションとした。

結果がyesになるべきデータとnoになるべきデータをおおよそ半々で学習する。
学習データの抽出は完全に割合以外はランダム。
Tnが増えてTpが減った。全体的にnoと返す傾向が高まったと考えられる。
ただしprecisionはFpが減ったために大幅に改善して見える。

```console
$ ./playground/14-validate_outtsv.py playground/tmp/out-hh64-t8-r0.001-b8-ab.tsv playground/tmp/splitted-ab.tsv
Aggregate: playground/tmp/out-hh64-t8-r0.001-b8-ab.tsv
  Tp=591 Tn=126 Fp=46 Fn=276
  accuracy:  0.6900866217516843
  precision: 0.9277864992150706
  recall:    0.6816608996539792
  F-measure: 0.7859042553191489
Aggregate: playground/tmp/splitted-ab.tsv
  Tp=652 Tn=57 Fp=115 Fn=215
  accuracy:  0.6823869104908566
  precision: 0.8500651890482399
  recall:    0.7520184544405998
  F-measure: 0.7980416156670748
```

半々ではnoが傾向が強すぎると仮定して割合を変えてみる。
結果がyesになるべきデータとnoになるべきデータを、
2:1 (ws64) および 1:2 (sw64) で学習する。

```console
$ ./playground/14-validate_outtsv.py playground/tmp/out-ws64-t8-r0.001-b8-ab.tsv playground/tmp/out-sw64-t8-r0.001-b8-ab.tsv playground/
tmp/splitted-ab.tsv
Aggregate: playground/tmp/out-ws64-t8-r0.001-b8-ab.tsv
  Tp=765 Tn=78 Fp=94 Fn=102
  accuracy:  0.8113570741097209
  precision: 0.8905704307334109
  recall:    0.8823529411764706
  F-measure: 0.8864426419466975
Aggregate: playground/tmp/out-sw64-t8-r0.001-b8-ab.tsv
  Tp=416 Tn=142 Fp=30 Fn=451
  accuracy:  0.5370548604427334
  precision: 0.9327354260089686
  recall:    0.4798154555940023
  F-measure: 0.6336633663366336
Aggregate: playground/tmp/splitted-ab.tsv
  Tp=652 Tn=57 Fp=115 Fn=215
  accuracy:  0.6823869104908566
  precision: 0.8500651890482399
  recall:    0.7520184544405998
  F-measure: 0.7980416156670748
```

2:1 (ws64) は Tp と Tnがともに増え、精度(precision)が若干改善している。
一方で 1:2 (sw64) ではTnは増え精度は改善しているが、
再現率は悪くなっており好ましくない。

なぜ 2:1 のバランスが良いかについては、あまりアイデアがない。
判断の閾値をほんの少しyes側に動かしている。
割合はその動かす距離と方向を決定している…くらいしか思いつかない。

## 普通の写真に対する効果

25枚の広告とは関係ない一般写真について、ws64での学習前後で推論し評価してみた。
データとしては全てyesとなるもの。

```console
./14-validate_outtsv.py ./tmp/out-ws64-t8-r0.001-b8-safe.tsv ./tmp/out-default-safe.tsv
Aggregate: ./tmp/out-ws64-t8-r0.001-b8-safe.tsv
  Tp=21 Tn=0 Fp=0 Fn=4
  accuracy:  0.84
  precision: 1.0
  recall:    0.84
  F-measure: 0.9130434782608696
Aggregate: ./tmp/out-default-safe.tsv
  Tp=15 Tn=0 Fp=0 Fn=10
  accuracy:  0.6
  precision: 1.0
  recall:    0.6
  F-measure: 0.7499999999999999
```

学習前は40%でnoと返していたのに対し、学習後は16%にまで減じている。
これはyesを多く返すように学習していることと符合している。
ただし内容を正しく判断したのだということはできない。

## ここまでのまとめ (2025-02-19)

* `yes` と答える圧力が強い (全体の85%以上が `yes` であることによる)
* 学習はイテレーションをした方が良い。過学習(常にyesと答える)が抑制される傾向にある
* ~学習データからtrue positiveを除いた方が `yes` 圧力は弱まる~
    * 実は学習データの yes と no の割合で決まってる
* 「true negative が減ってしまう」&「false positive が増えてしまう」ことが問題
    * true negative と false positive のデータだけで学習させたら、どうなるか見たほうが良い
    * no と答える圧力が高まり、全てに対して no と答えてしまう
* 10000件に対して 64 件の学習でも有意な学習結果が取れてしまう
    * より強く学習するには…学習件数を増やすのが良いだろうか?
* 母集団の統計的性質が大きく変わると、学習結果は役立たないのでは?
    * 普通の写真にたいしては上手く機能しない
