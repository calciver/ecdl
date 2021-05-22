import setuptools

setuptools.setup(
    name='ecdl',
    version='0.1',
    license='gpl-3.0',
    description='Elbow Classification Deep Learning',
    url='https://github.com/calciver/ecdl',
    keywords=['elbow-radiographs', 'machine-learning', 'CNN'],
    packages=setuptools.find_packages(),
    python_requires='>=3.6',
    classifiers=[
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Operating System :: OS Independent',
    ],
    install_requires=[
        'tensorflow',
    ]
)