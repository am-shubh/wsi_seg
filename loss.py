# Dependencies
from tensorflow.python.keras import backend as K
from tensorflow.python.framework import graph_io
import tensorflow as tf


class DiceLoss:
    def __init__(self, num_classes) -> None:
        self.num_classes = num_classes
        self.smooth = 1e-7

    # loss function
    def diceCoef(self, y_true, y_pred):
        # global num_classes

        # smooth = 1e-7

        if self.num_classes == 1:
            y_true_f = K.flatten(y_true)
            y_pred_f = K.flatten(y_pred)
            
        else:
            y_true_f = K.flatten(
                K.one_hot(K.cast(y_true, "int32"), num_classes=self.num_classes)[..., 1:]
            )
            y_pred_f = K.flatten(y_pred[..., 1:])

        intersection = K.sum(y_true_f * y_pred_f)
        return (2.0 * intersection + self.smooth) / (
            K.sum(y_true_f * y_true_f) + K.sum(y_pred_f * y_pred_f) + self.smooth
        )

        # intersect = K.sum(y_true_f * y_pred_f, axis=-1)
        # denom = K.sum(y_true_f + y_pred_f, axis=-1)
        # return K.mean((2. * intersect / (denom + smooth)))


    def diceCoefLoss(self, y_true, y_pred):
        return 1.0 - self.diceCoef(y_true, y_pred)


# freezing the model to .pb format
def freezeGraph(
    graph,
    session,
    output,
    save_pb_dir=".",
    save_pb_name="frozen_model.pb",
    save_pb_as_text=False,
):
    with graph.as_default():
        graphdef_inf = tf.graph_util.remove_training_nodes(graph.as_graph_def())
        graphdef_frozen = tf.graph_util.convert_variables_to_constants(
            session, graphdef_inf, output
        )
        graph_io.write_graph(
            graphdef_frozen, save_pb_dir, save_pb_name, as_text=save_pb_as_text
        )
        return graphdef_frozen

