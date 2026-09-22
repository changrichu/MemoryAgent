"""MemoryAgent - 基于多级记忆机制的 Agentic RAG 对话系统"""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = [
        line.strip() for line in f
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="memoryagent",
    version="0.1.0",
    description="基于多级记忆机制的 Agentic RAG 对话系统",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="MemoryAgent Contributors",
    license="MIT",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=8.0.0",
            "black>=24.0.0",
            "ruff>=0.3.0",
        ],
        "eval": [
            "ragas>=0.1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "memoryagent=ragagent.main:run",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
