# AILabs.tw 中文語音辨識系統:「雅婷Yating」

### 系統簡介：

採用深層類神經網路為聲學模型，搭配 N-gram 語言模型與 Weighted Finite State Transducer，期望提供即時且高準確率的雲端語音辨識。語料方面皆使用台灣節目、論壇的語音與文字資料，力求貼近在地的發音與用語習慣。

### 適用情境：
* 日常對話識別
* 自動語音逐字稿
* 自動新聞節目字幕

### 訓練資料來源：
* 台灣新聞
* 電視節目
* PTT文章
* AILabs.tw 自行蒐集之語料

### 性能指標:

中文字錯誤率(WER)：

* 中文語料測試集：8.93%
* 華語文聽力測驗模擬試題：10.39%

### 支援音訊格式：
16bit 16kHz PCM mono audio stream


### 通訊協定(Websocket)：
Gatekeeper endpoints： `wss://ime.ailabs.tw/ime/ws/1/`


client 透過 websocket 建立連線到 gatekeeper endpoint 後：

```
Client -> IME Gatekeeper
    {
        "action": "open_session",
        "pipeline": "ime"
    }

Client <- IME Gatekeeper
    {
        "auth_type": "basic",
        "auth_challenge": "... random string ..."
    }

Client -> IME Gatekeeper
    {
        "authorization:" "... sha1(key + ' ' + auth_challenge) in HEX ..."
    }

Client <- IME Gatekeeper (成功，可以開始傳送語音)
    {
        "status": "ok"
    }

Client <- IME Gatekeeper (失敗，回應後切斷連線)
    {
        "status": "error",
        "detail": "..."         # 詳細錯誤訊息
    }
```

傳送語音
client -> server

```
[PCM 16bit binary audio chunk]
[PCM 16bit binary audio chunk]
[PCM 16bit binary audio chunk]
...
[EOF: empty audio chunk]            # (optional) 送長度為 0 的 data chunk 來立刻結束句子
```

client <- server

```

{
    pipe: { "asr_state": "first_chunk_received" }       # 收到第一個 chunk 時回傳，不論那個 chunk 內容是什麼
}
{
    pipe: { "asr_state": "utterance_begin" }            # 偵測到語音開始時回傳
}
{
    pipe: { "asr_sentence": "金" }
}
{
    pipe: { "asr_sentence": "今天" }
}
{
    pipe: { "asr_sentence": "今天天" }
}
{
    pipe: { "asr_sentence": "今天天氣" }
}

...
# 有 asr_final: true 表⽰示⼀一個句句⼦子結束了了。
# asr_begin_time 及 asr_end_time 是這個句句⼦子的開始、結束時間 (秒為單位)，
# 以這個 connection 連上後傳上來來的第⼀一個 frame 為時間 0 開始算起。
{
    pipe: { "asr_sentence": "今天天氣很好", "asr_final": true,
               "asr_begin_time": 6.017, "asr_end_time": 7.271 }
}

# ASR decoding 結束
{
    pipe: { "asr_state": "utterance_end" }
}
# 若是最後因為收到EOF而結束，另外回傳告知client
{
    pipe: { "asr_eof": true }
}
# 若繼續送聲⾳音 chunks，則會接著看到 utterance_begin 表⽰示繼續 decode

```

### 範例程式碼:
* [ekko](../demos/ekko/)


<a name="api-tokens"></a>
### API Token 試用:

報名截止(7/20)之前，參賽者可以先利用以下 API Token 作測試，待正式報名錄取後，每一組參賽者會再額外發一組專屬 Token，請妥善保管。

* JmcREGWCEzfs3TgVQeF3

請欲報名之參賽者多加利用。
