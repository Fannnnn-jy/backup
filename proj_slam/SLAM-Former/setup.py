from setuptools import setup, find_packages

setup(
    name='slamformer',
    version='0.1.0',
    description='SLAM-Former: Putting SLAM into One Transformer.',
    packages=find_packages(include=['evals', 'evals.*', 'src/slamformer', 'src/slamformer.*']),
)
