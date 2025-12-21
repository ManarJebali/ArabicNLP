from setuptools import setup, find_packages

setup(
    name="arabic-nlp",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "torch>=1.10.0",
        "nltk>=3.6.0",
        "openpyxl>=3.0.0",
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="Arabic NLP with Word Embeddings",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3.8",
    ],
)