from setuptools import setup, find_packages

setup(
    name="frap_predictor",
    version="0.1.0",
    description="Paquete para predecir actividad antioxidante FRAP en residuos agroindustriales",
    author="TEAM",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "tensorflow>=2.6.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "shap>=0.40.0"
    ],
    python_requires=">=3.8",
)
