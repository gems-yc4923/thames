from setuptools import setup, find_packages

setup(
    name='yctmodel',
    version='1.1.0',
    packages=['yctmodel'],
    install_requires=['scikit-learn','pandas','matplotlib',
                        'numpy','scipy','seaborn','xgboost',],
    url='https://github.com/gems-yc4923/thames.git',
    license='MIT',
    author='yc4923'
)