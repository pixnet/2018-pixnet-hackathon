import platform
import subprocess as sp


if 'armv7' not in platform.platform():
    raise RuntimeError('Only run on raspberry pi 3 model B+')

try:
    p = sp.Popen(['mpg123', '--help'], stderr=sp.PIPE, stdout=sp.PIPE)
    retcode = p.wait()
except Exception as err:
    raise RuntimeError('Please install mpg123, `sudo apt install mpg123`')

try:
    import aiy  # noqa
except ImportError as err:
    raise RuntimeError(
        'Please install `aiy` package, which should be stored in /home/pi/AIY-projects-python/src/'
    )
