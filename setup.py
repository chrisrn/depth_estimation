from setuptools import setup, find_packages

setup(
    name='depth_estimation',
    version='1.0.0',
    description='Deep learning pipeline for depth estimation',
    install_requires=[r.replace('\n', '') for r in open('requirements.txt')],
    tests_require=[
        'nose',
        'flake8',
        'coverage~=4.5.1',
    ],
    setup_requires=[
        'nose',
        'flake8',
        'coverage~=4.5.1',
        'packaging'
    ],
    test_suite='nose.collector',
    packages=find_packages(),
    include_package_data=True
)
