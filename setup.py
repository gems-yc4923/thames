from setuptools import setup, find_packages

setup(
    name='yctmodel',
    version='5.2.16',
    packages=['yctmodel'],
    description="ModelSelector automates ensemble creation, added AutoTuner.",
    long_description_content_type="text/markdown",
    long_description=open("README.md").read(),
    install_requires=['scikit-learn','pandas','matplotlib',
                        'numpy','scipy','seaborn','xgboost',],
    url='https://github.com/gems-yc4923/thames.git',
    license='MIT',
    author='yc4923'
)