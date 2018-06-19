import subprocess as sp
import logging


logger = logging.getLogger('ekko')


class _Player:
    cmd_for_mp3 = ['mpg123-pulse', '-q', '-']
    cmd_for_wav = ['aplay', '-q', '-']

    def play_bytes(self, fd, _format='mp3'):
        if _format == 'mp3':
            cmd = self.cmd_for_mp3
        elif _format == 'wav':
            cmd = self.cmd_for_wav
        else:
            raise AssertionError('Only support mp3 or wav format.')

        player = sp.Popen(cmd, stdin=fd)
        retcode = player.wait()
        if retcode:
            logger.error('%s failed with %d', cmd[0], retcode)


simple_player = _Player()
