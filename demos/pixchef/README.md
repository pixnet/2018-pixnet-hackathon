# PIXCHEF
This is a demo implementation of "**Globally and Locally Consistent Image Completion**".

### Quick Start
ℹ️ We arranged all scripts to be executed as module.

1. To get the summaries/visuilizations of the models, run:

    `python -m models`

    The results will be stored in `graphs/`.

2. Place your data in `data/` folder, and start training the model:

    `python -m training`

    evaluation results will be placed in `evaluate/` folder.

3. `test_models.py` will output image processing results of some stages(layers) in the model pipeline for debugging purpose and giving some intuition.

    `python -m test_models`

    results will be placed in `temp/` folder.


### Model Architecture
![](./static/model_v2.png)

### Overall Graph Visualization
![](./static/glcic_graph.png)


## Reference
* [Globally and Locally Consistent Image Completion](http://hi.cs.waseda.ac.jp/~iizuka/projects/completion/en/)
