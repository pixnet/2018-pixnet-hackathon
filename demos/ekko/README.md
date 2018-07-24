# Ekko

Demo App for Voice Kit. Push the button, and say something, **ekko** will echo what you said by gTTS.

## Requirements

0. This demo were only tested on Raspberry Pi 3 Model B+ (armv7l from `uname -a`), if you use RPI with older version, some configs/implementations may not work functionally.

1. Download the [customized image](https://github.com/google/aiyprojects-raspbian/releases) for `AIY Kit`, and follow install guide from [here](https://www.raspberrypi.org/documentation/installation/installing-images/README.md).

2. After installation, boot the system and follow the guide from [here](https://github.com/google/aiyprojects-raspbian/blob/aiyprojects/HACKING.md), check the driver comfigurations.

⚠️ Google AIY project 仍不太穩定，本範例使用[aiyprojects-2018-04-13](https://github.com/google/aiyprojects-raspbian/releases/tag/v20180413)版本的 Image，部分官方腳本存在衝突，不過只要執行 `python aiyprojects-raspbian/checkpoints/check_audio.py` 可以正常聽到聲音與錄音，上述步驟可跳過，否則請確認 driver 是否正常安裝。

3. Install [`mpg123`](https://www.mpg123.de/) to handle mp3 playback.

    `sudo apt install mpg123`

### Python

This demo only works with python3.5+ since we need `async/await`.

### pip

1. Backup `/etc/pip.conf` and edit it as:

    ```
    [global]
    extra-index-url=https://www.piwheels.hostedpi.com/simple
    ```

    This additional index will find you the prebuilt packages when you install ones using `pip`, and make installation process faster! [[ref](https://www.raspberrypi.org/blog/piwheels/)]

2. Install dependencies:

```
$ pip3 install -r requirements.txt
```

⚠️ 如果您不習慣使用 virtualenv 作為 Python 環境管理工具，`pip3` 安裝請加 `sudo`，套件將安裝至系統環境(not recommended😢):

```
$ sudo pip3 install -r requirements.txt
```


## ℹ️ Quick Start

1. Prepare environment variables for AILabs' ASR service, save these informations in a file, e.g. `~/.secrets.env`:

    ```
    export WS_ENDPOINT=wss://ime.ailabs.tw/ime/ws/1/
    export WS_TOKEN=<PASTE-YOUR-API-TOKEN-HERE>
    ```
2. Download this repo:

    ```
    $ (cd ~/Desktop; git clone https://github.com/pixnet/2018-pixnet-hackathon)
    ```

3. Run this demo app:

    ```
    $ (cd ~/Desktop/2018-pixnet-hackathon/demos; source ~/.secrets.env; python3 -m ekko.main)
    ```

4. Push the button, and say something, **ekko** will echo what you said by `gTTS`.
