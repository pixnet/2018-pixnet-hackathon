import subprocess as sp
import os
import logging
import asyncio
from ._player import simple_player

logger = logging.getLogger('ekko')
PJ = os.path.join
this_dir = os.path.dirname(os.path.abspath(__file__))


class _Recorder:
    cmd_for_arecord = 'arecord --format=S16_LE --duration={duration} --rate={rate} --file-type=raw -q'
    cmd_kill_pulseaudio = 'pulseaudio --kill'

    def __init__(self, duration=4, rate=16000):
        self.duration = duration
        self.rate = rate
        self.cmd_for_arecord = self.cmd_for_arecord.format(duration=duration, rate=rate)

    async def record_wav(self, queue, kill=True):
        cmd = self.cmd_for_arecord
        process = sp.Popen(cmd.split(' '), stdout=sp.PIPE)
        with open(PJ(this_dir, 'res/activate.mp3'), 'rb') as fd:
            simple_player.play_bytes(fd, kill=False)

        while True:
            await asyncio.sleep(0.5)
            data = process.stdout.read(self.rate * 2)
            await queue.put(data)
            if len(data) == 0:
                break
        retcode = process.wait()

        if retcode:
            logger.error('%s failed with %d', cmd[0], retcode)

        if kill:
            p = sp.Popen(self.cmd_kill_pulseaudio.split(' '))
            p.wait()

        return retcode


simple_recorder = _Recorder()
