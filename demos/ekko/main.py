import asyncio
import tempfile
import websockets
import os
import json
import hashlib
import aiy.audio  # noqa # pylint: disable=import-error
from gtts import gTTS
import logging

from ._player import simple_player


logger = logging.getLogger('ekko')
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setFormatter(
    logging.Formatter(
        '%(asctime)s - %(levelname)s - %(name)s - %(module)s.%(lineno)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
)
logger.addHandler(ch)


PJ = os.path.join
this_dir = os.path.dirname(os.path.abspath(__file__))

ws_endpoint = os.environ.get('WS_ENDPOINT')
ws_token = os.environ.get('WS_TOKEN')

ws_queue = asyncio.Queue()
speaker_queue = asyncio.Queue()


if ws_endpoint is None or ws_token is None:
    raise ValueError('Must provide websocket')


class _BufferDump:
    def __init__(self, duration):
        self._buff = tempfile.TemporaryFile()
        self._bytes = 0
        self._bytes_limit = int(duration * 16000) * 1 * 2

    def add_data(self, data):
        max_bytes = self._bytes_limit - self._bytes
        data = data[: max_bytes]
        self._bytes += len(data)
        if data:
            self._buff.write(data)

    def is_done(self):
        return self._bytes >= self._bytes_limit

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self._buff.close()


async def ws_to_tts(lang='zh-tw'):
    sentence = await speaker_queue.get()
    tts = gTTS(sentence, lang=lang, lang_check=False)
    tempf = tempfile.TemporaryFile()
    tts.write_to_fp(tempf)
    tempf.seek(0)
    simple_player.play_bytes(tempf)
    return 0


async def record_to_buffer(duration=3):
    recorder = aiy.audio.get_recorder()
    dumper = _BufferDump(duration)

    data = None
    logger.info('Starting recording from microphone.')
    simple_player.play_bytes(
        open(PJ(this_dir, 'res/activate.mp3'), 'rb'),
        _format='mp3'
    )
    with recorder, dumper:
        recorder.add_processor(dumper)
        while not dumper.is_done():
            await asyncio.sleep(0.1)

        dumper._buff.seek(0)
        data = dumper._buff.read()

    logger.info('Recording is finished.')

    if data:
        await ws_queue.put(data)

        return 0
    else:
        return -1


async def handle_websocket():
    logger.info('Opening websocket to WS_HOST')
    async with websockets.connect(ws_endpoint) as ws:
        logger.info('Establishing websocket setups.')
        await ws.send(json.dumps(dict(action='open_session', pipeline='ime')))
        recv = await ws.recv()
        logger.info('Recieving credential from ebsocket')
        auth = ws_token + ' ' + json.loads(recv).get('auth_challenge')  # XXX
        hash_auth = hashlib.sha1(auth.encode('utf-8'))

        payload = dict(authorization=hash_auth.hexdigest())
        await ws.send(json.dumps(payload))

        recv = await ws.recv()
        logger.info('Finishing handshake: %s' % recv)
        logger.info('Ready to send PCM bytes.')

        data = await ws_queue.get()  # coroutine should be blocked here.

        logger.info('Calling speech recognition api service')
        await ws.send(data)
        await ws.send(b'')

        sentences = []

        while True:
            msg = await ws.recv()
            body = json.loads(msg)
            if 'asr_sentence' in body['pipe']:
                sentences.append(body['pipe'].get('asr_sentence'))
                continue
            is_eof = body['pipe'].get('asr_eof', False)
            if is_eof:
                break

        simple_player.play_bytes(
            open(PJ(this_dir, 'res/response.mp3'), 'rb'),
            _format='mp3'
        )

        out = sentences[-1] if len(sentences) > 0 else '蛤'
        await speaker_queue.put(out)


if __name__ == '__main__':

    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(
            asyncio.gather(record_to_buffer(),
                           handle_websocket(),
                           ws_to_tts())
        )
        simple_player.play_bytes(
            open(PJ(this_dir, 'res/deactivate.mp3'), 'rb'),
            _format='mp3'
        )
    finally:
        loop.close()

    logger.info('Done')
