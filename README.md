# Document

## 1. Environment

In the server, we create an environment called `VOC_A`.

```shell
conda create -n VOC_A python=3.8
```

## 2. Configure the file

In fold `configs` exist one file `Base-ResNet101.yaml`.

We define a python class `Config` in `config.py` to store the configuration parameters.