import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='clean-fid',
    version='0.1.7',
    author="Gaurav Parmar",
    author_email="gparmar@andrew.cmu.edu",
    description="FID calculation in PyTorch with proper image resizing and quantization steps",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/GaParmar/clean-fid",
    packages=['cleanfid'],
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
)