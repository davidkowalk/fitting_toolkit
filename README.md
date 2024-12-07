![GitHub commit activity](https://img.shields.io/github/commit-activity/m/davidkowalk/fitting_toolkit)
![GitHub License](https://img.shields.io/github/license/davidkowalk/fitting_toolkit)
![University](https://img.shields.io/badge/Univeristy_of_Bonn-brown)
![Version](https://img.shields.io/badge/version-1.0.1-green)
![GitHub Repo stars](https://img.shields.io/github/stars/davidkowalk/fitting_toolkit?style=flat&label=github%20stars)



# Fitting Toolkit
This toolkit aims at providing flexible and powerful tools for data analysis and modelling, but remain easy to use.

Here, I aim to strike a balance between the two extremes in this field. On one side are toolkits such as Kafe2, which prioritize ease of use and convenience but limit user control over the output, often resulting in highly specialized graphics that frequently do not meet standards required for publication without considerable effort. On the other side are data analysis systems like CERN's ROOT, which offer exceptional speed and capability but come with a steep learning curve and often exceed the requirements of most experiments.

This toolkit is aimed primarily at my peers, students of physics at the university of bonn, and to a degree at professionals within my field. I am optimizing this toolkit to be used on the scale typical of lab courses and homework assignments but if possible it should be powerful enough to run decently sized datasets on an average laptop.

This toolkit wraps numpy for fast data management and manipulation, scipy for `curve_fit()` and matplotlib for display options.

Check out the `docs` folder for documentation and tutorials.

## Quick Introduction

### Installation

There are multiple ways to install this package. The easiest is via pip:
```
pip install fitting-toolkit
```
If you need a specific version (for example due to compatibillity issues) you can specify the version via `fitting-toolkit==version`, e.g:
```
pip install fitting-toolkit==1.0.1
```

### Alternative Installation Methods

You can find all releases here: 

<a href= "https://github.com/davidkowalk/fitting_toolkit/releases"><svg width="180" height="48" viewBox="0 0 180 48" fill="none" xmlns="http://www.w3.org/2000/svg"><rect width="180" height="48" rx="4" fill="#00A2FF"/><path d="M28.59 21H27V16C27 15.45 26.55 15 26 15H22C21.45 15 21 15.45 21 16V21H19.41C18.52 21 18.07 22.08 18.7 22.71L23.29 27.3C23.68 27.69 24.31 27.69 24.7 27.3L29.29 22.71C29.92 22.08 29.48 21 28.59 21ZM17 31C17 31.55 17.45 32 18 32H30C30.55 32 31 31.55 31 31C31 30.45 30.55 30 30 30H18C17.45 30 17 30.45 17 31Z" fill="white"/><path d="M57.248 15.3195C59.024 15.3195 60.576 15.6635 61.904 16.3515C63.248 17.0235 64.28 17.9995 65 19.2795C65.736 20.5435 66.104 22.0235 66.104 23.7195C66.104 25.4155 65.736 26.8875 65 28.1355C64.28 29.3835 63.248 30.3435 61.904 31.0155C60.576 31.6715 59.024 31.9995 57.248 31.9995H51.8V15.3195H57.248ZM57.248 29.7675C59.2 29.7675 60.696 29.2395 61.736 28.1835C62.776 27.1275 63.296 25.6395 63.296 23.7195C63.296 21.7835 62.776 20.2715 61.736 19.1835C60.696 18.0955 59.2 17.5515 57.248 17.5515H54.536V29.7675H57.248ZM74.5516 32.2155C73.3036 32.2155 72.1756 31.9355 71.1676 31.3755C70.1596 30.7995 69.3676 29.9995 68.7916 28.9755C68.2156 27.9355 67.9276 26.7355 67.9276 25.3755C67.9276 24.0315 68.2236 22.8395 68.8156 21.7995C69.4076 20.7595 70.2156 19.9595 71.2396 19.3995C72.2636 18.8395 73.4076 18.5595 74.6716 18.5595C75.9356 18.5595 77.0796 18.8395 78.1036 19.3995C79.1276 19.9595 79.9356 20.7595 80.5276 21.7995C81.1196 22.8395 81.4156 24.0315 81.4156 25.3755C81.4156 26.7195 81.1116 27.9115 80.5036 28.9515C79.8956 29.9915 79.0636 30.7995 78.0076 31.3755C76.9676 31.9355 75.8156 32.2155 74.5516 32.2155ZM74.5516 29.8395C75.2556 29.8395 75.9116 29.6715 76.5196 29.3355C77.1436 28.9995 77.6476 28.4955 78.0316 27.8235C78.4156 27.1515 78.6076 26.3355 78.6076 25.3755C78.6076 24.4155 78.4236 23.6075 78.0556 22.9515C77.6876 22.2795 77.1996 21.7755 76.5916 21.4395C75.9836 21.1035 75.3276 20.9355 74.6236 20.9355C73.9196 20.9355 73.2636 21.1035 72.6556 21.4395C72.0636 21.7755 71.5916 22.2795 71.2396 22.9515C70.8876 23.6075 70.7116 24.4155 70.7116 25.3755C70.7116 26.7995 71.0716 27.9035 71.7916 28.6875C72.5276 29.4555 73.4476 29.8395 74.5516 29.8395ZM101.88 18.7755L97.7763 31.9995H94.8963L92.2323 22.2315L89.5683 31.9995H86.6883L82.5603 18.7755H85.3443L88.1043 29.4075L90.9123 18.7755H93.7683L96.4563 29.3595L99.1923 18.7755H101.88ZM110.597 18.5595C111.637 18.5595 112.565 18.7755 113.381 19.2075C114.213 19.6395 114.861 20.2795 115.325 21.1275C115.789 21.9755 116.021 22.9995 116.021 24.1995V31.9995H113.309V24.6075C113.309 23.4235 113.013 22.5195 112.421 21.8955C111.829 21.2555 111.021 20.9355 109.997 20.9355C108.973 20.9355 108.157 21.2555 107.549 21.8955C106.957 22.5195 106.661 23.4235 106.661 24.6075V31.9995H103.925V18.7755H106.661V20.2875C107.109 19.7435 107.677 19.3195 108.365 19.0155C109.069 18.7115 109.813 18.5595 110.597 18.5595ZM122.247 14.2395V31.9995H119.511V14.2395H122.247ZM131.575 32.2155C130.327 32.2155 129.199 31.9355 128.191 31.3755C127.183 30.7995 126.391 29.9995 125.815 28.9755C125.239 27.9355 124.951 26.7355 124.951 25.3755C124.951 24.0315 125.247 22.8395 125.839 21.7995C126.431 20.7595 127.239 19.9595 128.263 19.3995C129.287 18.8395 130.431 18.5595 131.695 18.5595C132.959 18.5595 134.103 18.8395 135.127 19.3995C136.151 19.9595 136.959 20.7595 137.551 21.7995C138.143 22.8395 138.439 24.0315 138.439 25.3755C138.439 26.7195 138.135 27.9115 137.527 28.9515C136.919 29.9915 136.087 30.7995 135.031 31.3755C133.991 31.9355 132.839 32.2155 131.575 32.2155ZM131.575 29.8395C132.279 29.8395 132.935 29.6715 133.543 29.3355C134.167 28.9995 134.671 28.4955 135.055 27.8235C135.439 27.1515 135.631 26.3355 135.631 25.3755C135.631 24.4155 135.447 23.6075 135.079 22.9515C134.711 22.2795 134.223 21.7755 133.615 21.4395C133.007 21.1035 132.351 20.9355 131.647 20.9355C130.943 20.9355 130.287 21.1035 129.679 21.4395C129.087 21.7755 128.615 22.2795 128.263 22.9515C127.911 23.6075 127.735 24.4155 127.735 25.3755C127.735 26.7995 128.095 27.9035 128.815 28.6875C129.551 29.4555 130.471 29.8395 131.575 29.8395ZM140.232 25.3275C140.232 23.9995 140.504 22.8235 141.048 21.7995C141.608 20.7755 142.36 19.9835 143.304 19.4235C144.264 18.8475 145.32 18.5595 146.472 18.5595C147.512 18.5595 148.416 18.7675 149.184 19.1835C149.968 19.5835 150.592 20.0875 151.056 20.6955V18.7755H153.816V31.9995H151.056V30.0315C150.592 30.6555 149.96 31.1755 149.16 31.5915C148.36 32.0075 147.448 32.2155 146.424 32.2155C145.288 32.2155 144.248 31.9275 143.304 31.3515C142.36 30.7595 141.608 29.9435 141.048 28.9035C140.504 27.8475 140.232 26.6555 140.232 25.3275ZM151.056 25.3755C151.056 24.4635 150.864 23.6715 150.48 22.9995C150.112 22.3275 149.624 21.8155 149.016 21.4635C148.408 21.1115 147.752 20.9355 147.048 20.9355C146.344 20.9355 145.688 21.1115 145.08 21.4635C144.472 21.7995 143.976 22.3035 143.592 22.9755C143.224 23.6315 143.04 24.4155 143.04 25.3275C143.04 26.2395 143.224 27.0395 143.592 27.7275C143.976 28.4155 144.472 28.9435 145.08 29.3115C145.704 29.6635 146.36 29.8395 147.048 29.8395C147.752 29.8395 148.408 29.6635 149.016 29.3115C149.624 28.9595 150.112 28.4475 150.48 27.7755C150.864 27.0875 151.056 26.2875 151.056 25.3755ZM156.497 25.3275C156.497 23.9995 156.769 22.8235 157.313 21.7995C157.873 20.7755 158.625 19.9835 159.569 19.4235C160.529 18.8475 161.593 18.5595 162.761 18.5595C163.625 18.5595 164.473 18.7515 165.305 19.1355C166.153 19.5035 166.825 19.9995 167.321 20.6235V14.2395H170.081V31.9995H167.321V30.0075C166.873 30.6475 166.249 31.1755 165.449 31.5915C164.665 32.0075 163.761 32.2155 162.737 32.2155C161.585 32.2155 160.529 31.9275 159.569 31.3515C158.625 30.7595 157.873 29.9435 157.313 28.9035C156.769 27.8475 156.497 26.6555 156.497 25.3275ZM167.321 25.3755C167.321 24.4635 167.129 23.6715 166.745 22.9995C166.377 22.3275 165.889 21.8155 165.281 21.4635C164.673 21.1115 164.017 20.9355 163.313 20.9355C162.609 20.9355 161.953 21.1115 161.345 21.4635C160.737 21.7995 160.241 22.3035 159.857 22.9755C159.489 23.6315 159.305 24.4155 159.305 25.3275C159.305 26.2395 159.489 27.0395 159.857 27.7275C160.241 28.4155 160.737 28.9435 161.345 29.3115C161.969 29.6635 162.625 29.8395 163.313 29.8395C164.017 29.8395 164.673 29.6635 165.281 29.3115C165.889 28.9595 166.377 28.4475 166.745 27.7755C167.129 27.0875 167.321 26.2875 167.321 25.3755Z" fill="white"/><rect x="50" y="35.9995" width="108" height="0.001" fill="white"/></svg></a>


After downloading the desired version you can find the `fitting_toolkit.py` in the `src` folder and copy it into your project.

To build the project yourself and install it, make sure `setuptools` and `wheel` are installed, then run
```
python3 setup.py sdist bdist_wheel
pip install .\dist\fitting_toolkit-1.0.1-py3-none-any.whl --force-reinstall   
pip show fitting-toolkit -v
```

### Requirements
This project requires the following modules along with their dependencies:
- numpy
- matplotlib
- scipy

It is highly recommended that the user familiarizes themselves with the functionality of these modules first. A rudimentary understanding of `numpy` and `matplotlib.pyplot` is required.

If you install via pip the dependencies will automatically be installed. However if the project files are used directly you may want to install dependencies manually:

To install the dependencies, first a [virtual environment](https://docs.python.org/3/library/venv.html) should be created. `requirements.txt` lists all necessary packages. Run:
```
pip install -r requirements.txt
```

### Getting Started

You can now import the relevant functions into your code:
```python
from fitting_toolkit import curve_fit, plot_fit 
import numpy as np
```
The `curve_fit` requires numpy-arrays. Therefore numpy has to be imported as well.

We can now start by simply defining our data.
```python
x = np.array((1, 2, 3, 4, 5))
y = np.array((1, 2, 1.75, 2.25, 3))
dy = 0.1*y+0.05
dx = 0.1
```
We chose a simple linear model:
```python
def f(x, a, b):
    return a * x + b
```
We can now fit the model to the data:
```python
params, cov, lower_conf, upper_conf = curve_fit(f, x, y, yerror=dy)
```
This functions returns 4 arrays. First the parameters of the model, the covariance matrix of those parameters and then the lower and upper limits of the confidence interval around the fit. Note that the confidence interval is absolute. To get the error in relation to the fitted function you would need to find the difference at each point.

The resulting fit can now be plotted. This toolkit provides a premade function to generate plots:
```python
from matplotlib import pyplot as plt
fig, ax = plot_fit(x, y, f, params, lower_conf, upper_conf, xerror=dx, yerror=dy)
plt.show()
```
Note that the fitted function is not automatically displayed. Instead the figure and axis-objects are returned.

![Example Graph](./docs/img/example_fit.png)

For a deeper explanation and tutorials please reference the [documentation](./docs/manual.md/).

## Literature:
[1] Vugrin, K. W., L. P. Swiler, R. M. Roberts, N. J. Stucky-Mack, and S. P. Sullivan (2007), Confidence region estimation techniques for nonlinear regression in groundwater flow: Three case studies, Water Resour. Res., 43, W03423, https://doi.org/10.1029/2005WR004804. \
[2] Dennis D. Boos. "Introduction to the Bootstrap World." Statist. Sci. 18 (2) 168 - 174, May 2003. https://doi.org/10.1214/ss/1063994971