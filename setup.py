from setuptools import setup, find_packages

setup(
    name='PackESAM',
    version='1.0',
    packages=find_packages(),
    include_package_data=True,  # 包含非代码文件
    description='A packaged version of Efficient SAM',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='William Kuang',
    author_email='dakuang2002@126.com',
    url='https://github.com/Ender-William',
    install_requires=[
        'onnxruntime-gpu >= 1.12.3',
        'numpy >= 1.24.2',
        'opencv-python >=4.1.1',
        'Pillow >= 7.1.2',
    ],
    classifiers=[
        'Intended Audience :: Developers',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
    ]
)
