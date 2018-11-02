import setuptools


def parse_requirements(requirements):
    with open(requirements) as f:
        return [l.strip('\n') for l in f if l.strip('\n') and not l.startswith('#')]

    
reqs = parse_requirements('requirements.txt')


setuptools.setup(
    name="trendspy",
    version="0.0.3",
    author="Dmitry E. Marienko",
    author_email="dmitry.ema@gmail.com",
    description="Collection of python utilities for timeseries trends analysis",
    long_description="Collection of python utilities for timeseries trends analysis",
    url="https://github.com/dmarienko/trendspy",
    packages=setuptools.find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    license='GPL',
    keywords=['trend', 'trends', 'timeseries', 'patterns'],
    download_url='https://github.com/dmarienko/trendspy/archive/v0.1.tar.gz',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Operating System :: OS Independent",
    ],
    install_requires=reqs,
)