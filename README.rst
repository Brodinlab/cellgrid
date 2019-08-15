========
CellGrid
========


.. image:: https://img.shields.io/pypi/v/cellgrid.svg
        :target: https://pypi.python.org/pypi/cellgrid


.. image:: https://travis-ci.org/Brodinlab/cellgrid.svg?branch=master
        :target: https://travis-ci.org/Brodinlab/cellgrid




Cell classification by learning known phenotypes


* Free software: MIT license
* Documentation: https://cellgrid.readthedocs.io.


Usage
=====

Train and use the classifier

.. code-block:: python

    from cellgrid.preprocessing import transform
    from cellgrid.ensemble import GridSchema, GridClassifier

    #transform the data
    df_x_train = transform(df_x_train)
    #load schema from the json file
    schema = GridSchema.from_json(path_to_schema)
    #train the classifier
    clf = GridClassifier(schema)
    clf.fit(df_x_train, df_y_train)

    #classify
    df_x = transform(df_x)
    df_y = clf.predict(df_x_test)

    #save and load the model
    from cellgrid.ensemble import save_model, load_model
    save_model(clf, path)
    clf = load_model(path)


Requirements
^^^^^^^^^^^^

df_x
""""

The single cell data in a a (pandas) data frame format.

df_y
""""

A data frame that contains the layered labels.
An example:

+-----------+---------+---------------------+
| layer1    | layer2  | layer3              |
+===========+=========+=====================+
| cells     | B       | Naive B             |
+-----------+---------+---------------------+
| cells     | B       | IgD+ Memory B       |
+-----------+---------+---------------------+
| cells     | CD4T    | Central memory CD4T |
+-----------+---------+---------------------+
| non-cells |         |                     |
+-----------+---------+---------------------+
| cells     | CD4T    |                     |
+-----------+---------+---------------------+

Schema
""""""
A definition of the layered structure.
An example:

.. code-block:: json

    [
        {
            "name": "all-events",
            "parent": null,
            "model_class_name": "xgb",
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
                "CD5",
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
                "CD7",
            ]
        }
    ]




Credits
=======


This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
