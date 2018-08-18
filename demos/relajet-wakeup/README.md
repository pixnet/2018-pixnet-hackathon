## Pixnet + Google AIY Kit Wake-up example

RelaJet 洞見未來這次提供參賽者可以用 Wake-up 方式叫醒 Google AIY Kit

## Demo video
[喚醒效果可跟天貓精靈做比較](https://youtu.be/3iTFkLpioR0)

## Python 版本
* 2.7

## Raspberry OS
* Rasbian

## 安裝麥克風的驅動程式
* `git clone https://github.com/shivasiddharth/GassistPi.git`
* `cd GassistPi`
* `sudo chmod +x ./audio-drivers/AIY-HAT/scripts/configure-driver.sh`
* `sudo ./audio-drivers/AIY-HAT/scripts/configure-driver.sh`
* `sudo reboot`
* `sudo chmod +x ./audio-drivers/AIY-HAT/scripts/install-alsa-config.sh`
* `sudo ./audio-drivers/AIY-HAT/scripts/install-alsa-config.sh`

### 安裝環境
* 首先, rpi 須先裝好 aubio, webrtcvad, pyaubio
* pip install aubio pyaubio webrtcvad
* 再來安裝 opencv-python, 不過注意一下不可用 `sudo apt-get install opencv-python` 安裝，因為目前在 python2.7 trunk 上的版本是 opencv 2.4，版本太舊，因此需要從新 build opencv 3.3.1 版本
* 另外也可以跟 `blue.chen@relajet.com` 寄信詢問已經全安裝好的 RPI image，直接安裝至 SD card 中， 省去上述各種安裝麻煩

## 下載主程式
* 下載 RelaJet Wake-up 包: `wget https://s3-ap-northeast-1.amazonaws.com/relajet/RELAJET_KWS.tar`
* 解壓縮: `tar -xvf ./RELAJET_KWS.tar`
* `cd KKBOX_KWS1`
* `cp relajet_deploy.prototxt.gpg /var/model/`
* `cp relajet_iter_60000.caffemodel.gpg /var/model/`

## 使用方式

### 客製化您專屬喚醒詞
* 請至 KKBOX_KWS1 資料夾下
* 輸入 `python ./vad1.pyc tmp`
* 出現這個 recording tmp #0: 字樣則可以開始講
* 請離麥克風有適當的距離
* 聲音音量正常講，不必過大也不必過小
* 喚醒詞長度時間盡量不小於 0.5 秒，不超過 2s
* 每講完一次會進行下一個循環，一共要講五次

### 訓練客製化喚醒詞
* `python ./train-tmp.pyc`
* 過程中請勿斷開，約需等待 20s - 30s

### 啟動程式
* `python ./tmp.pyc 0.6 2`
* 0.6 代表模型靈敏度 數值介在 0-1.0，越大代表靈敏度越高，相對容易喚醒
* 2   代表麥克風拾音參數， 2 最大，0 最小
* 講『洞見未來』時，畫面會出現 relajet
* 講『開燈』時，畫面會出現 on
* 講『關燈』時，畫面會出現 off
* 講您的客製化喚醒詞時，畫面會出現 tmp
* Enjoy !!!!
