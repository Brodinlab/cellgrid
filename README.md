# CellGrid

<a href="https://pypi.python.org/pypi/cellgrid">
<img src="https://img.shields.io/pypi/v/cellgrid.svg">
</a>
<a href="https://travis-ci.org/Brodinlab/cellgrid">
<img src="https://travis-ci.org/Brodinlab/cellgrid.svg?branch=master">
</a>

Cell classification by learning known phenotypes



## Install

```bash
$ pip install cellgrid
```


## Get Started
1. **Create a schema file**.   
Cellgrid trains a set of machine learning models in a hierarchical structure
in order to classify the cell populations in the same manner.
This schema is defined as a list in a json,
in which each element contains:
    * **name**.
    * **parent**. Name of the parent model.
    * **model_class_name**. Name of the The base model class. 
    The following options are supported:
        * xgb
        * random-forest
        * linear-regression
    * **markers**. The markers that are used for training the model.
    * For example:

    ```json
    [
        {
            "name": "all-events",
            "parent": null,
            "model_class_name": "random-forest",
            "markers": [
                "Ce140Di",
                "Ir191Di"
            ]
        },
        {
            "name": "cells",
            "parent": "all-events",
            "model_class_name": "xgb",
            "markers": [
                "CD45",
                "HLA-ABC",
                "CD57",
                "CD19",
                "CD5"
            ]
        },
        {
            "name": "CD4T",
            "parent": "cells",
            "model_class_name": "xgb",
            "markers": [
                "CD5",
                "CD4",
                "CD8a",
                "CD31",
                "CD25",
                "CD3e",
                "CD7"
            ]
        }
    ]
    ```
1. **Train a GridClassifier**.   
    ```python
    from cellgrid.preprocessing import transform
    from cellgrid.ensemble import GridSchema, GridClassifier    
    
    #load schema from the json file
    schema = GridSchema.from_json(path_to_schema)
    #transform the data
    x_train = transform(x_train)
    #train the classifier
    clf = GridClassifier(schema)
    clf.fit(x_train, y_train)
    ```
1. **Score**. Return the F1 score of every model.
   ```python
   x_test = transform(x_test)
   clf.score(x_test, y_test)
   ```   
1. **Predict**
   ```python
   x = transform(x)
   y = clf.predict(x)
   ```
1. **Save and load**
   ```python
   from cellgrid.ensemble import save_model, load_model
    
   save_model(clf, path)
   clf = load_model(path)
   ``` 

## API
### GridClassifier
#### Constructor
```python
GridClassifier(schema)
```
###### Arguments
* schema: See above regarding to schema definition.

#### Methods
##### fit
```python
fit(x_train, y_train)
```
Train the classifier

###### Arguments
* x_train: The single cell dataset. 
* y_train: The labels in a hierarchical structure.
An example:


| layer1    | layer2   | layer3  |
| --------- |:--------:| ------: |
| cells     | B       | Naive B             |
| cells     | B       | IgD+ Memory B       |
| cells     | CD4T    | Central memory CD4T |
| non-cells |         |                     |
| cells     | CD4T    |                     |

##### predict
```python
predict(x)
```
Predict the hierarchical labels for dataset ```x```

##### score
```python
score(x_test, y_test)
```
Return F1 scores of every model. 

### GridSchema
#### Methods
##### from_json
```python
from_json(filepath=None)
```
Load the schema from a json file.
See above regarding to schema definition.


## License
MIT

## Credits



This package was created with 
<a href="https://github.com/audreyr/cookiecutter">
 Cookiecutter
</a> 
and the
<a href="https://github.com/audreyr/cookiecutter-pypackage"> 
`audreyr/cookiecutter-pypackage`
</a>
 project template.
