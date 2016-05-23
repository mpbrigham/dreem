# encoding: utf-8
"""Set Matplotlib defaults.
"""

from __future__ import division
from matplotlib import ticker
import matplotlib.pyplot as plt
from palettable.tableau import TableauLight_10
from cycler import cycler



def plot_clean(yticker=3, xticker=4):
    plt.gcf()
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().get_xaxis().tick_bottom()
    plt.gca().get_yaxis().tick_left()
    plt.gca().yaxis.set_major_locator(ticker.MaxNLocator(yticker))
    plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(xticker))