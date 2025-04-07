import fitting_toolkit as ft
from matplotlib import pyplot as plt 
import numpy as np

def model(x, a, b):
    return a*x + b

def get_data():
    dU = 0.1
    ds = 0.5

    U = np.asarray([5.0, 7.0, 10.0, 12.0])
    s = np.asarray([31.43, 21.16, 14.23, 11.5])

    return U, dU, s, ds


def main():
    U, dU, s, ds = get_data()

    f = 10/s
    df = 10*ds/s**2

    fit = ft.curve_fit(model, U, f, yerror=df, model_resolution=100, nsigma = 3)
    fig, ax = ft.plot_fit(U, f, xerror=dU, yerror=df, fit=fit)
    ax.set_xlabel("Motorspannung / $U$")
    ax.set_ylabel("Frequenz / $\\nu$")

    ft.utils.to_pgf(fig, "./img/plot.pgf")

if __name__ == "__main__":
    main()