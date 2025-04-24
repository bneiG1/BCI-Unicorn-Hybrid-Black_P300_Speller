from setuptools import setup, find_packages

setup(
    name="bci_p300_speller",
    version="0.1.0",
    description="Real-time P300 Speller BCI system for Unicorn Hybrid Black EEG device.",
    author="Your Name",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "mne",
        "scikit-learn",
        "PyWavelets",
        "matplotlib",
        "seaborn",
        "PyQt5",
        "joblib"
    ],
    python_requires=">=3.8",
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "generate_sample_data=generate_sample_data:generate_sample_eeg_dataset",
        ],
    },
)
