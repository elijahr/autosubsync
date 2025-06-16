from setuptools import setup
from os import path

# Get the long description from the DESCRIPTION file
with open("DESCRIPTION.rst") as f:
    long_description = f.read()

package_name = "autosubsync"

setup(
    name=package_name,
    version="1.0.1",
    description="Automatically synchronize subtitles with audio",
    long_description=long_description,
    url="https://github.com/oseiskar/" + package_name,
    author="Otto Seiskari",
    license="MIT",
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        "Development Status :: 5 - Production/Stable",
        "Environment :: Console",
        "Operating System :: POSIX",
        "Intended Audience :: End Users/Desktop",
        "Topic :: Multimedia",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 3",
    ],
    keywords="subtitles syncrhonization srt ffmpeg",
    packages=["autosubsync"],
    # distribute the trained model file in the package
    package_data={package_name: ["../trained-model.bin"]},
    # define command line entry point
    entry_points={"console_scripts": ["%s=%s.main:cli_packaged" % (package_name, package_name)]},
    install_requires=["numpy", "soundfile"],
    extras_require={
        # dev requirements are needed for training the model
        "dev": ["pandas", "scikit-learn"]
    },
)
