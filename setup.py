from setuptools import setup, find_packages

# Read dependencies from requirements.txt
with open('requirements.txt') as f:
    install_requires = [line.strip() for line in f if line.strip() and not line.startswith('#')]

# Read version from version.txt
with open('version.txt') as f:
    version = f.read().strip()

setup(
    name="cvs",
    version=version,
    packages=['cvs'] + ['cvs.' + pkg for pkg in find_packages(where='cvs')],
    package_dir={'cvs': 'cvs'},
    install_requires=install_requires,
    entry_points={
        'console_scripts': [
            'cvs=cvs.main:main',
        ],
    },
    include_package_data=True,
    description="Cluster validation suite for AI training readiness",
    author="Advanced Micro Devices, Inc.",
    author_email="support@amd.com",
    url="https://github.com/ROCm/cvs",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: System Administrators",
        "Intended Audience :: Developers",
        "Topic :: System :: Clustering",
        "Topic :: Software Development :: Testing",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
