# Ekko

Demo App for Voice Kit. Push the button, and say something, **ekko** will echo what you said by gTTS.

## Requirements

1. Download the [customized image](https://github.com/google/aiyprojects-raspbian/releases) for `AIY Kit`, and follow install guide from [here](https://www.raspberrypi.org/documentation/installation/installing-images/README.md).

2. After installation, boot the system and follow the guide from [here](https://github.com/google/aiyprojects-raspbian/blob/aiyprojects/HACKING.md), check the driver comfigurations.


### Python

This demo only works with python3.5 since we need `async/await`.

### pip

1. Backup `/etc/pip.conf` and edit it as:
    ```
    [global]
    extra-index-url=https://www.piwheels.hostedpi.com/simple
    ```
    This additional index will find you the prebuilt packages when you install ones using `pip`. [[ref](https://www.raspberrypi.org/blog/piwheels/)]

2. Install dependencies:

```
$ pip install -r requirements.txt
```

## ℹ️ Quick Start

1. Prepare environment variables for AILabs' ASR service, save these informations in a file, e.g. `~/.secrets.env`:

    ```
    export WS_ENDPOINT=wss://ime.ailabs.tw/ime/ws/1/
    export WS_TOKEN=<PASTE-YOUR-API-TOKEN-HERE>
    ```
2. Download this repo:

    ```
    $ cd ~/Desktop
    $ git clone https://github.com/pixnet/2018-pixnet-hackathon
    ```

3. Run this demo app:

    ```
    $ (cd ~/Desktop/2018-pixnet-hackathon/demos; source ~/.secrets.env; python -m ekko.main)
    ```

4. Push the button, and say something, **ekko** will echo what you said by `gTTS`.
