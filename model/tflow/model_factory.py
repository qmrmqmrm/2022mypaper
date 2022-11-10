import tensorflow as tf
import numpy as np
from tensorflow import keras

import model.framework.backbone as back
import model.framework.head as head
import model.framework.neck as neck
import model.framework.decoder as decoder
import utils.framework.util_function as uf
import config as cfg
import model.framework.model_util as mu


class ModelFactory:
    def __init__(self, batch_size, input_shape, anchors_per_scale,
                 backbone_name=cfg.Architecture.BACKBONE,
                 neck_name=cfg.Architecture.NECK,
                 head_name=cfg.Architecture.HEAD,
                 backbone_conv_args=cfg.Architecture.BACKBONE_CONV_ARGS,
                 neck_conv_args=cfg.Architecture.NECK_CONV_ARGS,
                 head_conv_args=cfg.Architecture.HEAD_CONV_ARGS,
                 num_anchors_per_scale=cfg.ModelOutput.NUM_ANCHORS_PER_SCALE,
                 head_composition=cfg.ModelOutput.HEAD_COMPOSITION,
                 num_lane_anchors_per_scale=cfg.ModelOutput.NUM_LANE_ANCHORS_PER_SCALE,
                 out_channels=cfg.ModelOutput.NUM_MAIN_CHANNELS,
                 lane_out_channels=cfg.ModelOutput.NUM_LANE_CHANNELS,
                 training=True
                 ):
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.anchors_per_scale = anchors_per_scale
        self.backbone_name = backbone_name
        self.neck_name = neck_name
        self.head_name = head_name
        self.backbone_conv_args = backbone_conv_args
        self.neck_conv_args = neck_conv_args
        self.head_conv_args = head_conv_args
        self.num_anchors_per_scale = num_anchors_per_scale
        self.num_lane_anchors_per_scale = num_lane_anchors_per_scale
        self.head_composition = head_composition
        self.out_channels = out_channels
        self.lane_out_channels = lane_out_channels
        self.training = training
        mu.CustomConv2D.CALL_COUNT = -1
        print(f"[ModelFactory] batch size={batch_size}, input shape={input_shape}")
        print(f"[ModelFactory] backbone={self.backbone_name}, NECK={self.neck_name} ,HEAD={self.head_name}")

    def get_model(self):
        backbone_model = back.backbone_factory(self.backbone_name, self.backbone_conv_args, self.training)
        neck_model = neck.neck_factory(self.neck_name, self.neck_conv_args, self.training,
                                       self.num_anchors_per_scale, self.out_channels,
                                       self.num_lane_anchors_per_scale, self.lane_out_channels)
        head_model = head.head_factory(self.head_name, self.head_conv_args, self.training, self.num_anchors_per_scale,
                                       self.head_composition, self.num_lane_anchors_per_scale, self.lane_out_channels)

        input_tensor = tf.keras.layers.Input(shape=self.input_shape, batch_size=self.batch_size)
        backbone_features = backbone_model(input_tensor)
        neck_features = neck_model(backbone_features)
        head_features = head_model(neck_features)

        output_features = {}
        feature_decoder = decoder.FeatureDecoder(self.anchors_per_scale)
        feat_box_logit = head_features[:-1] if cfg.ModelOutput.LANE_DET else head_features
        feat_box_logit = uf.merge_and_slice_features(feat_box_logit, False, "feat_box")
        output_features["feat_box"] = feature_decoder(feat_box_logit)

        if cfg.ModelOutput.LANE_DET:
            lane_decoder = decoder.FeatureLaneDecoder()
            output_features["feat_lane_logit"] = uf.merge_and_slice_features(head_features[-1:], False, "feat_lane")
            output_features["feat_lane"] = lane_decoder(output_features["feat_lane_logit"])

        for key in output_features.keys():
            for slice_key in output_features[key].keys():
                if slice_key != "whole":
                    for scale_index in range(len(output_features[key][slice_key])):
                        output_features[key][slice_key][scale_index] = uf.merge_dim_hwa(
                            output_features[key][slice_key][scale_index])

        yolo_model = tf.keras.Model(inputs=input_tensor, outputs=output_features, name="yolo_model")
        return yolo_model


# ==================================================


def test_model_factory():
    print("===== start test_model_factory")
    anchors = [np.array([[10, 20], [30, 40], [50, 60]], dtype=np.float32),
               np.array([[10, 20], [30, 40], [50, 60]], dtype=np.float32),
               np.array([[10, 20], [30, 40], [50, 60]], dtype=np.float32),
               ]
    batch_size = 1
    imshape = (512, 1280, 3)
    model = ModelFactory(batch_size, imshape, anchors).get_model()
    input_tensor = tf.zeros((batch_size, 512, 1280, 3))
    # print(model.summary())
    keras.utils.plot_model(model, to_file='Efficient_test.png', show_shapes=True)
    output = model(input_tensor)
    print("print output key and tensor shape")
    uf.print_structure("output", output)

    print("!!! test_model_factory passed !!!")


if __name__ == "__main__":
    uf.set_gpu_configs()
    test_model_factory()
