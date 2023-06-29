from setuptools import find_packages, setup

__version__ = "1.5.2"


setup(
    name='mpnn',
    packages=find_packages(),
    package_data={'mpnn': ['py.typed']},
    entry_points={
        'console_scripts': [
            'mpnn_train=mpnn.train:mpnn_train',
            'mpnn_predict=mpnn.train:mpnn_predict',
            'mpnn_fingerprint=mpnn.train:mpnn_fingerprint',
            'mpnn_hyperopt=mpnn.hyperparameter_optimization:mpnn_hyperopt',
            'mpnn_interpret=mpnn.interpret:mpnn_interpret',
            'mpnn_web=mpnn.web.run:mpnn_web',
            'sklearn_train=mpnn.sklearn_train:sklearn_train',
            'sklearn_predict=mpnn.sklearn_predict:sklearn_predict',
        ]
    },
    install_requires=[
        'flask>=1.1.2',
        'hyperopt>=0.2.3',
        'matplotlib>=3.1.3',
        'numpy>=1.18.1',
        'pandas>=1.0.3',
        'pandas-flavor>=0.2.0',
        'scikit-learn>=0.22.2.post1',
        'scipy>=1.5.2',
        'sphinx>=3.1.2',
        'tensorboardX>=2.0',
        'torch>=1.5.2',
        'tqdm>=4.45.0',
        'typed-argument-parser>=1.6.1',
        'rdkit>=2020.03.1.0',
        'descriptastorus'
    ],
    extras_require={
        'test': [
            'pytest>=6.2.2',
            'parameterized>=0.8.1'
        ]
    },
    python_requires='>=3.7',
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Operating System :: OS Independent'
    ],
    keywords=[
        'chemistry',
        'machine learning',
        'property prediction',
        'message passing neural network',
        'graph neural network'
    ]
)
