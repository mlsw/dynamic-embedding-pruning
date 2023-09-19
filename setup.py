from setuptools import setup, find_namespace_packages

setup(
    name="dynamic_embedding_pruning",
    version="0.0.1",
    package_dir={"": "src"},
    packages=find_namespace_packages(where="src"),
    python_requires=">=3.11.0",
    install_requires=[
        "transformers[accelerate]>=4.29.1",
        "torch>=2.0.1",
        "datasets>=2.12.0",
    ],
)
