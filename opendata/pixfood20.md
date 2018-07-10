# PIXFOOD20

**`PIXFOOD20`** 是痞客邦利用深度學習訓練出美食分類器，並針對熱門美食文章中的圖片進行預測所挑選出 **20+2** 個種類的美食圖庫，準確率超過 90%(validation accuracy)，分類包含食物類與非食物類別：

食物類別，共計20類：
```
火鍋、牛排、咖啡、丼飯、滷肉飯、
生魚片、鬆餅、麵包、蛋糕、義大利麵、
牛肉麵、小籠包、生菜沙拉、拉麵、串燒、
壽司、漢堡、薯條、冰淇淋、手搖飲料
```

非食物類別，共計2類：
```
環境、菜單
```

## 資料集與選題

**`PIXFOOD20`** 資料集被切分為**訓練集**(Training set)與**測試集**(Testing set)，訓練集開放給所有參賽者下載，用來訓練深度學習的模型。

⚠️ **實際比賽題目只會由測試集中的20個美食分類抽出，非食物類別不在選題範圍內**。

## 資料範例與欄位說明

訓練集資料格式為 `jsonline`，每張圖片只有一個分類標籤，每種食物分類中圖片數量約有 `1500` 張，總計超過 30,000 張圖片，資料集大小約 `2.2 G`，資料範例如下：

```
{
  "blog_url": "http://ksdelicacy.pixnet.net/blog/post/66179970",
  "blog_url_hash": "036b8094e28ba2d58a7f1f9faf45818fdca00384",
  "image_url": "https://pic.pimg.tw/ksdelicacy/1509541252-1409835141.jpg",
  "image_path": "沙拉/食物/1509541252-1409835141.jpg",
  "tag": "沙拉",
  "score": 0.9632999897003174
}
{
  "blog_url": "http://asih.pixnet.net/blog/post/47824176",
  "blog_url_hash": "329b0d7dbe050c541b40e8fc6b959c170276c823",
  "image_url": "https://pic.pimg.tw/asih/1517508267-477228321.jpg",
  "image_path": "牛肉麵/食物/1517508267-477228321.jpg",
  "tag": "牛肉麵",
  "score": 1
}
{
  "blog_url": "http://flyblog.cc/blog/post/46977640",
  "blog_url_hash": "b8c2c094c8ce15e19ca2a7cb2f52fd8c9b9ce1b6",
  "image_url": "http://img.fun-life.com.tw/taipei/hotpot/DSC06794.JPG",
  "image_path": "牛肉麵/環境/dsc06794.jpg",
  "tag": "環境",
  "score": 0.9901000261306763
}
{
  "blog_url": "http://colonel466.pixnet.net/blog/post/117517822",
  "blog_url_hash": "3e38cc57360130383824b272ca797e4c4fec92f0",
  "image_url": "https://pic.pimg.tw/colonel466/1502721381-2841920820_n.jpg",
  "image_path": "蛋糕/菜單/1502721381-2841920820_n.jpg",
  "tag": "菜單",
  "score": 0.991599977016449
}
{
  "blog_url": "http://atwanted.pixnet.net/blog/post/223246366",
  "blog_url_hash": "1da1d5dd96359bbb9c13d7b18c73b9b80aed92a4",
  "image_url": "https://pic.pimg.tw/atwanted/1501065848-358304464_n.png",
  "image_path": "鬆餅/食物/1501065848-358304464_n.png",
  "tag": "鬆餅",
  "score": 1
}
```

欄位             |意義
----------------|:------------------
`blog_url`      | 圖片所在的部落格文章連結
`blog_url_hash` | 文章連結 hash 值
`image_url`     | 圖片的原始連結
`image_path`    | 下載資料集中實際的圖片路徑
`tag`           | 圖片的分類標籤
`score`         | 模型所預測的信心分數



## 資料下載

### 使用授權
若您下載下方連結所提供的資料集 (Dataset)，表示您同意以下的資料使用授權：

您可以：
* 自由應用提供的資料集，產生新的程式、文件、圖表等著作。
* 自由修改提供的資料集，產生衍生的資料集。

您必須：
* 在您的著作或衍生資料集明確標示資料來源與此份說明文件的連結。

您不可以：
* 使用可能混淆或困擾的商標或名稱。
* 造成痞客邦會員產生違反痞客邦會員條款之行為。
* 違反中華民國法令或造成第三人發生違反中華民國法令的行為。
* 另為提供他人資料集下載。亦即，您不可以複製一份資料集到您自己的網路服務上供他人下載，但您可以提供他人此份說明文件的連結。
* 如您利用提供的資料集，開發任何妨礙善良風俗之違法服務或程式工具，PIXNET 並不為此負任何法律連帶責任。

### 下載連結

* ⏬[PIXFOOD20](https://drive.google.com/open?id=1_NzFmdAK5VmzzmggX8voPbQGPFaFKAbF)

