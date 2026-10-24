from setuptools import find_packages, setup

__lib_name__ = "spacraft"
__lib_version__ = "1.0.0"
__description__ = "SpaCRAFT: spatial chromatin accessibility representation learning"

setup(
    name = __lib_name__,
    version = __lib_version__,
    description = __description__,
    package_dir={"": "src"},
    packages=find_packages("src"),
    entry_points={
        "console_scripts": [
            "spacraft=spacraft.cli:main",
        ]
    },
    zip_safe = False,
    include_package_data = True,
)
