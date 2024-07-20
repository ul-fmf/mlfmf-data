# MLFMF: Data Sets for Machine Learning for Mathematical Formalization

This is the repository that contains all the code that was used in our paper [MLFMF: Data Sets for Machine Learning for Mathematical Formalization](https://arxiv.org/abs/2310.16005)

```
@misc{bauer2023mlfmf,
      title={MLFMF: Data Sets for Machine Learning for Mathematical Formalization}, 
      author={Andrej Bauer and Matej Petković and Ljupčo Todorovski},
      year={2023},
      eprint={2310.16005},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

With it, we can 

- preprocess data (previously obtained from different repositories of Agda and Lean libraries repositories on GitHub and transformed to s-expressions),
- learn the models on the preprocessed data.

The data is available at [Zenodo](https://zenodo.org/records/10041075).


# The repository structure

- Directory `apaa` (a proof assistant assistant) contains all the sources.
- Directory `dumps` contains all the results of the experiments.
- Other python scripts are described below.


# Setting up the environment

Please, use the `requirements.txt` to install all the Python packages.
The easiest way to do so is by creating a virtual environment.
Open the command line and execute the following commands:

```
> python -m venv venv
> venv\Scripts\activate
(venv) > pip install -r requirements.txt
```

The activation command is different on Linux and Mac OS: use `source venv/bin/activate` instead.

# Data format

For easier re-use of data, we created a _lightweight_ format for storing the data sets.
Every data set corresponds to a single library from Lean or Agda and is stored in a separate folder
that contains


- `network.csv` file (which contains the network structure of the library),
- directory `entries` with many `.dag` files (which contain the abstract syntax trees of the entries in the library).

Instead of `entries`, `entries.zip` may be present. The main script will unzip it automatically˛

_Note: abstract syntax trees are guaranteed to be trees only for Agda libraries. In Lean - due to node sharing - they might be directed acyclic graphs (DAGs)._

## Network file

The `network.csv` file contains the rows of the following format:

- `node<tab><node name><tab><node properties>`,
- `link<tab><source node name><tab><sink node name><tab><link type><tab><link properties>`,

for example 

```
node    Relation.Nullary.Decidable.Core {'label': ':module'}
```

and 

```
link    Agda.Builtin.Bool.Bool.true 10  Agda.Builtin.Bool.Bool 6    REFERENCE_TYPE  {'w': 1}
```

## DAG files

Every dag file lists the nodes of the "abstract syntax tree" of a single entry in the library.
A node is represented by a single line, containing

- node id (a number that uniquely identifies the node in the file),
- node type (a string that identifies the type of the node),
- node description (a possibly empty string that contains an additional description of the node), and
- children ids (a possibly empty list of node ids that are children of the node).

For example,

```
NODE ID NODE TYPE   NODE DESCRIPTION    CHILDREN IDS
...
205 :type       [202, 204]
204 :apply      [203]
203 :name   "Agda.Builtin.Char.Char 6"  []
...
```

Note: In theory, every `.dag` file could contain node ids from 1 to N, where N is the number of nodes in the file. They are only used to reconstruct the definition into a tree or DAG.

## Internal data format

As mentioned above, the library currently uses a bit _heavier_ data structures. Instead of classes
`Entry` and `EntryNode` (found in `light_weight_data_loader.py`) which have only the necessary fields (like `children`, `type` etc.), more complex data structures are used (see `apaa.data.structures`). These have more fields (mostly properties that
are needed often due to the current design of the implemented models). (In the future, the internal data format will get
deprecated.)

For this reason, we provide a script that converts the _light-weight_ data format into the _internal_ data format: `light_to_heave_data.py`.

Note: Lean Mathlib 4 library needs more than 64 GB of memory to be fully loaded. For this reason, we provide
an optimized version for loading that only loads nodes of type `:entry`, its children and the nodes of type `:name`,
since only these are essential to reconstruct the network later. Omitting most of the nodes does not harm the performance
of the currently best working `node2vec` model, since it does not use any information from the definitions that are not already
present in the network.


## Running the scripts

### Data conversion
If you want to convert the _ligth-weight_ format into the _internal_ format, run `light_to_heavy_data.py` script:

Change the value of `lib_loc`(ation) and set the `optimized` parameter (`False` by default) to `True`
if you are trying to load Lean Mathlib 4 library:

```
if __name__ == "__main__":
    lib_loc = "path/to/lib/folder"
    convert_to_heavy_data(lib_loc, optimized=True)
```

**For large libraries, this might take a while!**


If you obtained the s-expressions by yourself, then, you can use `library_preprocessing.py` to

- extract the entries from s-expressions,
- create DAG structures,
- create a knowledge graph,
- create a data set with a fixed train/test split.


### Learning

If you want to learn various models, run `main_learner.py` script. In the main block,
change the library path and turn on/off the models:

```
if __name__ == "__main__":
    learn_recommender_models(
        "path/to/library",
        dummy=True,
        bow=True,
        tfidf=True,
        word_embedding=True,   # fastText
        analogies=True,
        node_to_vec=True,
        p_def_to_keep=0.1,
        force=False,
    )
    LOGGER.info("Done")
```

#### Results

The results should appear in the `dumps/experiments` folder.
For every run, there should be two files:

- meta file: contains model class, model parameters and learning time,
- prediction file: contains evaluation measures and the top recommendations.
