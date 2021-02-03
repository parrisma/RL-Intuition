from enum import Enum, unique

@unique
class Names(str, Enum):
    x_train = "x_train"
    x_test = "x_test"
    y_train = "y_train"
    y_test = "y_test"
    predictions_interim = "prediction_interim"
    predictions = "predictions"
    loss = "loss"
    learning_rate = "learning rate"
    num_epoch = "num epoch"
    batch_size = "batch size"
    interim = "interim"
    q_scale = "q_scale"