from setuptools import setup, find_packages


with open("cluster_experiment_utils/version.py") as f:
    exec(f.read())
    version = locals()["__version__"]


def get_descriptions():
    with open("README.md") as f:
        readme_content = f.read()

    return "Cluster Experiment Utils", readme_content


def get_requirements(file_path):
    with open(file_path) as f:
        __requirements = []
        for line in f.read().splitlines():
            if not line.startswith("#"):
                __requirements.append(line)
    return __requirements


requirements = get_requirements("requirements.txt")

keywords = ["experiment-utils", "experiment_utils"]

short_description, long_description = get_descriptions()

setup(
    name="cluster_experiment_utils",
    version=version,
    license="MIT",
    author="Renan Souza",
    author_email="contact@renansouza.org",
    description=short_description,
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/renan-souza/experiment_utils",
    include_package_data=True,
    install_requires=requirements,
    # extras_require=extras_require,
    packages=find_packages(),
    keywords=keywords,
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Natural Language :: English",
        # "Topic :: Documentation :: Sphinx",
        "Topic :: System :: Distributed Computing",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: System :: Logging",
        "Topic :: System :: Monitoring",
        "Topic :: Database",
    ],
    python_requires=">=3.8",
)
