import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


class RoptLogger(object):
    def __init__(self, name: str='Optimizer') -> None:
        self.name = name
        self.log = []

    @property
    def n_iter(self) -> int:
        return len(self.log)

    def writelog(self, gn: float, val: float, time: float) -> None:
        '''
        Append the information of the iterations.
        '''
        self.log.append([gn, val, time])

    def to_csv(self, csv_dir: str='./output') -> None:
        if not os.path.isdir(csv_dir):
            os.makedirs(csv_dir)
        df = pd.DataFrame(self.log, columns=['norm', 'value', 'time'])
        df.to_csv(os.path.join(csv_dir, self.name + '.csv'), index=None, header=None)


def rlog_show(results: list[RoptLogger], yscale: str='log') -> None:
    for result in results:
        x = range(result.n_iter)
        y = [row[0] for row in result.log]
        plt.plot(x, y, label=result.name)

    plt.legend()
    plt.yscale(yscale)
    plt.grid(which='major')
    plt.show()
