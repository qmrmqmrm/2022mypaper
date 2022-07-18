import numpy as np
import torch
import model.framework.backbone as back
import model.framework.neck as neck
import model.framework.head as head
import model.framework.decoder as decoder
import config as cfg
import model.framework.model_util as mu
import utils.framework.util_function as uf


class ModelFactory:
    def __init__(self, batch_size, input_shape, anchors_per_scale,
                 backbone_name=cfg.Architecture.BACKBONE,
                 neck_name=cfg.Architecture.NECK,
                 head_name=cfg.Architecture.HEAD,
                 backbone_conv_args=cfg.Architecture.BACKBONE_CONV_ARGS,
                 neck_conv_args=cfg.Architecture.NECK_CONV_ARGS,
                 head_conv_args=cfg.Architecture.HEAD_CONV_ARGS,
                 num_anchors_per_scale=cfg.ModelOutput.NUM_ANCHORS_PER_SCALE,
                 pred_composition=cfg.ModelOutput.PRED_HEAD_COMPOSITION,
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
        self.pred_composition = pred_composition
        self.out_channels = out_channels
        self.lane_out_channels = lane_out_channels
        self.training = training
        print(f"[ModelFactory] batch size={batch_size}, input shape={input_shape}")
        print(f"[ModelFactory] backbone={self.backbone_name}, NECK={self.neck_name} ,HEAD={self.head_name}")

    def get_model(self):
        backbone_model = back.backbone_factory(self.backbone_name, self.backbone_conv_args, self.training)
        neck_model = neck.neck_factory(self.neck_name, self.neck_conv_args, backbone_model.out_channels)
        head_model = head.head_factory(self.head_name, self.head_conv_args, neck_model.out_channels,
                                       self.num_anchors_per_scale, self.pred_composition)
        decode_features = decoder.FeatureDecoder(self.anchors_per_scale)
        model = torch.nn.Sequential(backbone_model, neck_model, head_model, decode_features)
        return model


def test_model_factory():
    anchors = [np.array([[10, 20], [30, 40], [50, 60]], dtype=np.float32),
               np.array([[10, 20], [30, 40], [50, 60]], dtype=np.float32),
               np.array([[10, 20], [30, 40], [50, 60]], dtype=np.float32),
               ]
    batch_size = 1
    imshape = (512, 1280, 3)
    input_test = torch.Tensor(np.zeros((1, 3, 512, 1280)))
    model = ModelFactory(batch_size, imshape, anchors).get_model()
    output = model(input_test)
    uf.print_structure("output", output)
    # make_dot(output, params=dict(model.named_parameters()), show_attrs=True, show_saved=True).render(
    #     "rnn_torchviz", format="png")


if __name__ == "__main__":
    test_model_factory()
