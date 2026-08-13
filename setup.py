from setuptools import setup, find_packages

setup(
    name="agriergo",
    version="0.1.0",
    description="Video-Based Farm Worker Ergonomics & Drudgery Assessment Platform",
    author="AgriErgo Team",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "ultralytics>=8.2.0",
        "opencv-python-headless>=4.8.0",
        "numpy>=1.24.0",
        "scipy>=1.11.0",
        "fastapi>=0.104.0",
        "uvicorn[standard]>=0.24.0",
        "python-multipart>=0.0.6",
        "aiofiles>=23.2.0",
        "pydantic>=2.5.0",
        "pandas>=2.1.0",
        "streamlit>=1.28.0",
        "plotly>=5.18.0",
        "tqdm>=4.66.0",
    ],
)
