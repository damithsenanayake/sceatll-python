from setuptools import setup, find_packages

setup(
    name='SCEATLL-PYTHON',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        # List any dependencies your package needs
    ],
    #  entry_points={
    #     'console_scripts': [
    #         'sceatll_command = Sceatll.sceatll_integrate:main',
    #     ],
    # },
    author='Damith Senanayake',
    author_email='dracusds123@gmail.com',
    description='A package to integrate single cell rna-seq data',
    url='https://github.com/damithsenanayake/sceatll-python',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)
