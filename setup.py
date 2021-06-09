import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='clean-fid',
    version='0.1.13',
    author="Gaurav Parmar",
    author_email="gparmar@andrew.cmu.edu",
    description="FID calculation in PyTorch with proper image resizing and quantization steps",
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=[
        "torch>=0.4.0",
        "torchvision>=0.2.1",
        "numpy>=1.14.3",
        "scipy>=1.0.1",
        "tqdm>=4.28.1",
        "pillow>=8.1",
        "requests==2.25.1"
    ],
    url="https://github.com/GaParmar/clean-fid",
    packages=['cleanfid'],
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
)