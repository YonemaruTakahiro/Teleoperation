"""
Glove sampling Rate
===================
A simple console application that displays the glove sampling rate on the
screen. The rate is estimated using the most recent n=200 readings.
"""

import collections
import time
import sys
import argparse

from cyberglove import CyberGlove


class CyberGloveRate(CyberGlove):
    def __init__(self, n_df, s_port=None, baud_rate=115200,
                 samples_per_read=1, cal_path=None, n=1):
        super(CyberGloveRate, self).__init__(n_df, s_port, baud_rate,
                                        samples_per_read, cal_path)
        self.times = collections.deque(maxlen=n)
        self.last_time = None
        self.n = int(n)

    def start(self):
        super(CyberGloveRate, self).start()
        self.main_loop()

    @property
    def rate(self):
        if not self.times:
            return 0.0
        else:
            return 1.0 / (sum(self.times) / float(self.n))

    def main_loop(self):
        while True:
            data = self.read()
            print(f'data:{data}')
            t = time.perf_counter()
            if self.last_time is not None:
                self.times.append(t - self.last_time)

            self.last_time = t
            print("\rEMG Rate: {}".format(self.rate), end='')
            sys.stdout.flush()






if __name__ == '__main__':
    cgr = CyberGloveRate(n_df=22, s_port="/dev/ttyUSB0", samples_per_read=1, n=200)
    cgr.start()