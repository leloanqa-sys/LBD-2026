from setuptools import setup, find_packages

setup(
    name="dapagliflozin-lbd",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.10",
    install_requires=[
        "pydantic>=2.0.0",
        "pyyaml>=6.0",
        "requests>=2.31.0",
        "networkx>=3.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "jsonschema>=4.18.0",
        "tqdm>=4.66.0",
        "python-dotenv>=1.0.0",
        "groq>=0.9.0"
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.7.0",
            "ruff>=0.1.0"
        ]
    }
)
