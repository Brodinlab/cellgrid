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


## Usage
1. Create a schema file. Cellgrid trains a set of machine learning models  
to classify the cell populations in a tree structure.
It requires to define the "Nodes" in a schema file. 
Each node contains:
    * name: Name of the node
    * parent: The parent node
    * model_class_name: The base model class for the node. 
    It has the following options:
        * xgb
        * random-forest
        * linear-regression
    * markers: The markers that are used for training the node.
    * An example:
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
2. Train a GridClassifier.  
    ```python
    from cellgrid.preprocessing import transform
    from cellgrid.ensemble import GridSchema, GridClassifier    
    
    #transform the data
    x_train = transform(x_train)
    #load schema from the json file
    schema = GridSchema.from_json(path_to_schema)
    #train the classifier
    clf = GridClassifier(schema)
    clf.fit(x_train, y_train)
    ```
3. use the classifer
   ```python
   x = transform(x)
   y = clf.predict(x)
   ```
4. save and load
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

#### Methods
##### fit
```python
fit(x_train, y_train)
```
Train the classifier

###### Arguments
* x_train: The single cell dataset. 
* y_train: The layered labels.
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
Predict the layered labels for dataset ```x```

##### score
```python
score(x_test, y_test)
```
Return F1 scores of every layer. 

### GridSchema
#### Methods
##### from_json
```python
from_json(filepath=None)
```
Load the schema from a json file 


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
