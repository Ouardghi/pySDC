import numpy as np

from pathlib import Path

import matplotlib.animation as animation

import matplotlib.pyplot as plt


def main():

    # path = 'data_NonLinSolv_N=4_dt=0.01_PC=MIN-SR-FLEX/'

    # compraison
    # turek3 = np.loadtxt("data_FEATFLOW/draglift_q2_cn_lv1-6_dt4/bdforces_lv3")
    turek4 = np.loadtxt("data_FEATFLOW/draglift_q2_cn_lv1-6_dt4/bdforces_lv4")

    LD = np.loadtxt("data_NonLinSolv_N=4_dt=0.01_PC=MIN-SR-FLEX/Liftdrag.txt")

    fig = plt.figure(2, figsize=(16, 13))

    ax = fig.add_subplot(211)
    plt.plot(LD[:, 0], LD[:, 1], color='k', ls='--')
    plt.plot(
        turek4[1:, 1],
        turek4[1:, 4],
        marker="x",
        markevery=50,
        linestyle="-",
        markersize=4,
        label="FEATFLOW (42016 dofs)",
    )
    ax.set_xlabel('Times')
    ax.set_ylabel('Lift')
    ax.set_xlim(0, 8)

    ax = fig.add_subplot(212)
    plt.plot(LD[:, 0], LD[:, 2], color='k', ls='--')
    plt.plot(
        turek4[1:, 1],
        turek4[1:, 3],
        marker="x",
        markevery=50,
        linestyle="-",
        markersize=4,
        label="FEATFLOW (42016 dofs)",
    )
    ax.set_xlabel('Times')
    ax.set_ylabel('Drag')
    ax.set_xlim(0, 8)


if __name__ == '__main__':
    main()
    plt.show()
