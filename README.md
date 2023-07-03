# Pipelines
This repo is created to run and combine different machine learning pipelines with other tasks that involve resource allocation.
* component initializes a model or resource and provides destructor as `__del__` to release it.
* pipes all have detailed instructions that uses one or more components.
## setup
Follow the setup in the main repo if pipeline is used a submodule.
```
pip install requirements.txt
pip install tensorflow
```
