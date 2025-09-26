from setuptools import setup, find_packages

setup(
    name='hytoolpy',
    version='0.1.0',
    description='Modélisation des essais de pompage hydrauliques (HyTool en Python)',
    author='Celestin Dartigues',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'pandas',
        'matplotlib',
        'ipywidgets',
        'openpyxl'
    ],
    include_package_data=True,
    zip_safe=False,
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8'
)
