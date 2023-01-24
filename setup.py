import io
import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def get_long_description():
    readme_file = os.path.join(os.path.dirname(__file__), "README.md")
    with io.open(readme_file, "r", encoding="utf-8") as f:
        return f.read()


def get_requirements():
    req_file = os.path.join(os.path.dirname(__file__), "requirements.txt")
    with io.open(req_file, "r", encoding="utf-8") as f:
        return [line.strip() for line in f]


requirements = get_requirements()
long_description = get_long_description()


setup(
    name='torch_edit_distance',
    version="0.4.0",
    description="PyTorch edit-distance functions",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/1ytic/pytorch-edit-distance",
    author="Ivan Sorokin",
    author_email="i.sorok1n@icloud.com",
    license="MIT",
    ext_modules=[
        CUDAExtension('torch_edit_distance_cuda', [
            'binding.cpp',
            'edit-distance.cu',
        ])
    ],
    packages=['torch_edit_distance'],
    cmdclass={
        'build_ext': BuildExtension
    },
    setup_requires=requirements,
    install_requires=requirements,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ])
