from dataclasses import dataclass
from typing import Any

# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from numpy import log10

pd.options.mode.chained_assignment = None
ISOCHRONE_API = "https://astromancer.skynet.unc.edu/api/cluster/isochrone"

filter_wavelengths = {
    "U": 0.364,
    "B": 0.442,
    "V": 0.54,
    "R": 0.647,
    "I": 0.7865,
    "u'": 0.354,
    "g'": 0.475,
    "r'": 0.622,
    "i'": 0.763,
    "z'": 0.905,
    "J": 1.25,
    "H": 1.65,
    "K": 2.15,
    "Ks": 2.15,
    "W1": 3.4,
    "W2": 4.6,
    "W3": 12,
    "W4": 22,
    "BP": 0.532,
    "G": 0.673,
    "RP": 0.797,
}


def calculate_lambda(A_v, R_v, filter_lambda=10**-6):
    x = (filter_lambda / 1) ** -1
    y = x - 1.82

    if 0.3 < x < 1.1:
        a = 0.574 * x**1.61
        b = -0.527 * x**1.61
    elif 1.1 < x < 3.3:
        a = (
            1
            + 0.17699 * y
            - 0.50447 * y**2
            - 0.02427 * y**3
            + 0.72085 * y**4
            + 0.01979 * y**5
            - 0.7753 * y**6
            + 0.32999 * y**7
        )
        b = (
            1.41338 * y
            + 2.28305 * y**2
            + 1.07233 * y**3
            - 5.38434 * y**4
            - 0.62251 * y**5
            + 5.3026 * y**6
            - 2.09002 * y**7
        )
    else:
        a = 0
        b = 0

    return A_v * (a + b / R_v)


@dataclass
class Isochrone:
    distance: float  # kpc
    log_age: float  # log10(years)
    metallicity: float  # solar
    b_vreddening: float  # E(B-V)
    blue_filter: str
    red_filter: str
    lum_filter: str

    def __post_init__(self):
        self.reddening = self.b_vreddening * 3.1
        self.update_data()

        self.blue_error = f"{self.blue_filter}_error"
        self.red_error = f"{self.red_filter}_error"
        self.lum_error = f"{self.lum_filter}_error"

    def __setattr__(self, name: str, value: Any):
        super().__setattr__(name, value)
        if name in {
            "log_age",
            "metallicity",
            "blue_filter",
            "red_filter",
            "lum_filter",
        }:
            self.update_data()

    def update_data(self):
        try:
            iso = requests.get(
                ISOCHRONE_API,
                params={
                    "age": self.log_age,
                    "metallicity": self.metallicity,
                    "blue_filter": self.blue_filter,
                    "red_filter": self.red_filter,
                    "lum_filter": self.lum_filter,
                },
            )
            self.data = np.array(iso.json()["data"])
        except AttributeError:
            # fails when initializing for some reason
            self.data = np.array([])


def find_residuals(stars: pd.DataFrame, iso: Isochrone):
    for filter in {iso.blue_filter, iso.red_filter, iso.lum_filter}:
        stars[filter] -= calculate_lambda(
            iso.reddening, 3.1, filter_wavelengths[filter]
        )

    # this is slightly redundant but also slightly faster
    stars = stars[
        stars[iso.blue_filter].notna()
        & stars[iso.red_filter].notna()
        & stars[iso.lum_filter].notna()
    ]
    stars[iso.lum_filter] -= 5 * log10(iso.distance * 10**3) - 5

    filter_errors = stars[[iso.blue_error, iso.red_error, iso.lum_error]].values
    max_errors = np.max(filter_errors, axis=0)
    min_errors = np.min(filter_errors, axis=0)
    normed_errors = 1 - np.linalg.norm(
        ((filter_errors - min_errors) / max_errors) * np.sqrt(1 / 3), axis=1
    )

    b_r = np.array(stars[iso.blue_filter] - stars[iso.red_filter])
    lum = np.array(stars[iso.lum_filter])
    cmd = np.stack([b_r, lum], axis=1)
    residuals = np.min(np.linalg.norm(cmd[:, None] - iso.data, axis=2), axis=1)

    # plt.scatter(stars[iso.blue_filter] - stars[iso.red_filter], stars[iso.lum_filter])
    # plt.scatter(b_r, lum)
    # plt.scatter(*zip(*iso.data), c="r")
    # plt.gca().invert_yaxis()
    # plt.show()

    return (residuals / np.max(residuals)) * normed_errors


if __name__ == "__main__":
    isochrone = Isochrone(
        distance=0.98,
        log_age=8.55,
        metallicity=-0.05,
        b_vreddening=0.28,
        blue_filter="BP",
        red_filter="RP",
        lum_filter="G",
    )

    stars = pd.read_csv("n.csv")
    # stars = pd.read_csv("a.csv")
    residuals = find_residuals(stars, isochrone)
    print(np.mean(residuals))
