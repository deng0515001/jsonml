#!/usr/bin/env python
# coding=utf-8
from setuptools import setup, find_packages

#打包命令 python3 setup.py bdist_wheel

setup(
    name="laxin",
    version="0.0.8",
    keywords=("laxin", "model pipeline"),
    description="The model pipeline for laxin group",
    long_description="The model pipeline for laxin group",
    license="MIT",

    url="",
    author="Peter Deng",
    author_email="451408963@qq.com",

    # package_dir={'lxtest': 'lxtest', 'utils': 'utils'},         # 指定哪些包的文件被映射到哪个源码包
    packages=['laxin'],       # 需要打包的目录。如果多个的话，可以使用find_packages()自动发现
    include_package_data=True,
    py_modules=[],          # 需要打包的python文件列表
    data_files=['laxin/udf_param.json'],          # 打包时需要打包的数据文件
    platforms="any",
    install_requires=[      # 需要安装的依赖包
        'sklearn>=0.0',
        'numpy>=0.0',
        'pandas>=0.0',
        'xgboost>=0.7'
    ],
    scripts=[],             # 安装时复制到PATH路径的脚本文件
    entry_points={
        'console_scripts': [    # 配置生成命令行工具及入口
            'Lx.shell = lx:shell',
            'Lx.web = lx:web'
        ]
    },
    classifiers=[           # 程序的所属分类列表
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    zip_safe=False
)
