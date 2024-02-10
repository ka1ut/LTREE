# LTREE
## 必要なライブラリをインストール
```
pip install -r requirements.txt
```

## 実行・動作
### main.pyの動作
1. `python -u "~/LTREE/main.py"`

2. 任意の文章を入力
```
Enter a long text: いいね！何か特別な予定があるの？

---------sentences---------
いいね！
何か特別な予定があるの？
---------------------------

-----------word------------
text: いい, label: 4
text: ね, label: 4
text: ！, label: 4
text: 何, label: 4
text: か, label: 4
text: 特別, label: 4
text: な, label: 4
text: 予定, label: 4
text: が, label: 4
text: ある, label: 4
text: の, label: 4
text: ？, label: 4
---------------------------

-----similarity_matrix-----
[[1.         0.50741903 0.50857748 0.44125827 0.39679835 0.1165102
  0.50633748 0.15192    0.28035095 0.         0.21073764 0.50857748]
 [0.50741903 1.         0.50846875 0.71059179 0.76593613 0.30426262
  0.85836622 0.36379552 0.73047743 0.         0.58978347 0.50846875]
 [0.50857748 0.50846875 1.         0.38751138 0.36681118 0.18337995
  0.46535324 0.23919275 0.29903834 0.         0.2467808  1.        ]
 [0.44125827 0.71059179 0.38751138 1.         0.84939944 0.32704981
  0.78547757 0.30481902 0.7060174  0.         0.58043839 0.38751138]
 [0.39679835 0.76593613 0.36681118 0.84939944 1.         0.26270343
  0.85392657 0.36948429 0.75967068 0.         0.66101234 0.36681118]
 [0.1165102  0.30426262 0.18337995 0.32704981 0.26270343 1.
  0.26085851 0.35435687 0.33444577 0.         0.38664029 0.18337995]
 [0.50633748 0.85836622 0.46535324 0.78547757 0.85392657 0.26085851
  1.         0.34482084 0.6496364  0.         0.51146114 0.46535324]
 [0.15192    0.36379552 0.23919275 0.30481902 0.36948429 0.35435687
  0.34482084 1.         0.40112662 0.         0.40345671 0.23919275]
 [0.28035095 0.73047743 0.29903834 0.7060174  0.75967068 0.33444577
  0.6496364  0.40112662 1.         0.         0.84797621 0.29903834]
 [0.         0.         0.         0.         0.         0.
  0.         0.         0.         0.         0.         0.        ]
 [0.21073764 0.58978347 0.2467808  0.58043839 0.66101234 0.38664029
  0.51146114 0.40345671 0.84797621 0.         1.         0.2467808 ]
 [0.50857748 0.50846875 1.         0.38751138 0.36681118 0.18337995
  0.46535324 0.23919275 0.29903834 0.         0.2467808  1.        ]]
```

### plot.pyの動作
1. `python -u "~/LTREE/plot.py"`
2. text_vectors.dbに保存された単語ベクトルが表示されます
   - plotの色の違い(Label)は、それぞれの文章を表す
<img width="978" alt="image" src="https://github.com/ka1ut/LTREE/assets/108340480/e3248a3e-675c-4e10-8214-3fc13abf38ef">

