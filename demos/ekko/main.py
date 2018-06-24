import asyncio
import tempfile
import websockets
import time
import os
import ujson as json
import hashlib
import aiy.audio  # noqa # pylint: disable=import-error
import aiy.voicehat  # noqa # pylint: disable=import-error
from gtts import gTTS
from ._player import simple_player
from ._recorder import simple_recorder
from .logging import logger
from .exceptions import WebSocketAuthenticationError


PJ = os.path.join
this_dir = os.path.dirname(os.path.abspath(__file__))

ws_endpoint = os.environ.get('WS_ENDPOINT')
ws_token = os.environ.get('WS_TOKEN')

led = aiy.voicehat.get_led()


if ws_endpoint is None or ws_token is None:
    raise ValueError('Must provide websocket')


async def ws_to_tts(speaker_queue, lang='zh-tw'):
    try:
        sentence = await speaker_queue.get()
    except Exception:
        return 1
    logger.info('Calling gTTS')
    tts = gTTS(sentence, lang=lang, lang_check=False)
    led.set_state(led.ON)
    with tempfile.TemporaryFile() as tempf:
        tts.write_to_fp(tempf)
        tempf.seek(0)
        logger.info('Speaking...')
        simple_player.play_bytes(tempf)
    led.set_state(led.OFF)
    return 0


async def record_to_buffer(ws_queue):
    logger.info('Recording from microphone.')
    retcode = await simple_recorder.record_wav(ws_queue)
    logger.info('Recording is finished.')
    logger.info('Calling ASR service')
    return retcode


async def handle_websocket(ws_queue, speaker_queue):
    logger.info('Opening websocket to WS_HOST')
    async with websockets.connect(ws_endpoint) as ws:
        logger.info('Checking authentication')
        await ws.send(json.dumps(dict(action='open_session', pipeline='ime')))
        recv = await ws.recv()
        auth = ws_token + ' ' + json.loads(recv).get('auth_challenge')  # XXX
        hash_auth = hashlib.sha1(auth.encode('utf-8'))

        payload = dict(authorization=hash_auth.hexdigest())
        await ws.send(json.dumps(payload))

        recv = await ws.recv()
        logger.info('Finishing websocket setup: %s' % recv)
        if json.loads(recv).get('status') != 'ok':
            raise WebSocketAuthenticationError('The websocket api token may not be available.')
        logger.info('Ready to send PCM bytes to WS_HOST.')

        async def wait_for_queue():
            while True:
                data = await ws_queue.get()
                await ws.send(data)
                if len(data) == 0:
                    break

        async def wait_for_ws():
            sentences = []
            while True:
                msg = await ws.recv()
                body = json.loads(msg)
                # logger.info(msg)
                if 'asr_sentence' in body['pipe']:
                    sentences.append(body['pipe'].get('asr_sentence'))
                    continue
                if body['pipe'].get('asr_eof', False):
                    break
            return sentences[-1] if len(sentences) > 0 else '蛤'

        asyncio.ensure_future(wait_for_queue())
        out = await asyncio.ensure_future(wait_for_ws())
        with open(PJ(this_dir, 'res/response.mp3'), 'rb') as fd:
            simple_player.play_bytes(fd)
        await speaker_queue.put(out)
        logger.info('STT result: %s' % out)


def main():

    led.set_state(led.PULSE_QUICK)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    ws_queue = asyncio.Queue()
    speaker_queue = asyncio.Queue()

    try:
        loop.run_until_complete(
            asyncio.gather(record_to_buffer(ws_queue),
                           handle_websocket(ws_queue, speaker_queue),
                           ws_to_tts(speaker_queue))
        )
        with open(PJ(this_dir, 'res/deactivate.mp3'), 'rb') as fd:
            simple_player.play_bytes(fd)

    except Exception as err:
        logger.error(err)
        loop.stop()
        loop.run_forever()

    finally:
        loop.close()
        led.set_state(led.OFF)
    logger.info('Done')


if __name__ == '__main__':
    with open(PJ(this_dir, 'res/init.mp3'), 'rb') as fd:
        simple_player.play_bytes(fd)
    led.set_state(led.BLINK_3)
    time.sleep(2)
    led.set_state(led.OFF)

    button = aiy.voicehat.get_button()
    button.on_press(main)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass

    finally:
        led.set_state(led.OFF)
        led.stop()
