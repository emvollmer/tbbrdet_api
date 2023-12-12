#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Selectable options to train and test a model.
A platform user can enter information in fields as defined by these options.

Based on: K Alibabaei's fasterrcnn_pytorch_api.git
https://git.scc.kit.edu/m-team/ai/fasterrcnn_pytorch_api/-/blob/master/fasterrcnn_pytorch_api/fields.py
"""
from webargs import validate
from marshmallow import Schema, fields, validates_schema, ValidationError
from tbbrdet_api import configs
from tbbrdet_api.misc import (
    ls_folders, get_weights_folder
)


class TrainArgsSchema(Schema):
    """
    Class of all selectable options to train the model
    """

    class Meta:
        ordered = True

    # backbones = fields.Str(
    #     required=True,
    #     metadata={
    #         'enum': configs.BACKBONES,      # currently resnet 50
    #         "description": "Model backbone options."
    #     }
    # )
    dataset_path = fields.String(
        metadata={
            "description": "Path to the dataset. If none is provided, "
                           f"the local '{configs.DATA_PATH}' folder will be "
                           "searched.\nIf connected, you can also use the "
                           f"remote folder '{configs.REMOTE_DATA_PATH}'.",
        },
        required=False,
        load_default=configs.DATA_PATH
    )

    architecture = fields.Str(
        load_default='swin',
        validate=validate.OneOf(configs.ARCHITECTURES),
        metadata={
            'description': 'Model architecture options.'
        }
    )

    train_from = fields.Str(
        required=True,
        metadata={
            'enum': (configs.TRAIN_OPTIONS
                     + ls_folders(configs.MODEL_PATH)
                     + ls_folders(configs.REMOTE_MODEL_PATH)),
            'description': 'Options for training model: from scratch, '
                           'from pretrained weights (transfer learning), or '
                           'resume the training of a previously trained '
                           'model by selecting the appropriate '
                           '(remote or local) model folder.'
        }
    )

    device = fields.Bool(
        load_default=True,
        metadata={
            'enum': [True, False],
            'description': "Computation/training device. The default is a GPU."
                           "Training won't work without a GPU!"
        }
    )

    epochs = fields.Int(
        load_default=4,
        metadata={'description': 'Number of epochs to train.'}
    )

    workers = fields.Int(
        load_default=2,
        metadata={
            'description': 'Number of workers for data processing / training.'}
    )

    batch = fields.Int(
        load_default=1,
        metadata={'description': 'Batch size to load the data.'}
    )

    lr = fields.Float(
        load_default=0.0001,
        metadata={'description': 'Learning rate.'}
    )

    seed = fields.Int(
        load_default=1,
        metadata={'description': 'Global seed number for training.'}
    )

    eval = fields.Str(
        load_default="bbox",
        metadata={
            'enum': ["bbox", "segm"],
            'description': "Evaluate performance according to bounding box "
                           "(object detection model) "
                           "or segmented area (instance segmentation model)"
        }
    )

    @validates_schema
    def validate_required_fields(self, data):
        if data['device'] is False:
            # NOTE: this does not work!
            raise ValidationError(
                'Training requires a GPU. Please obtain one before continuing.'
            )

        if data['train_from'] == 'coco':
            # NOTE: this does not work!
            if not get_weights_folder(data).is_dir():
                raise ValidationError(
                    f"No pretrained weights folder for {data['architecture']}."
                    f" No training with {data['train_from']} weights with this"
                    f" architecture possible!"
                    f" Please select a different architecture."
                )


class PredictArgsSchema(Schema):
    """
    Class of all selectable options to test / predict with a model
    """

    class Meta:
        ordered = True

    input = fields.Field(
        required=True,
        metadata={
            'type': "file",
            'location': "form",
            'description': 'Input an image.'
        }
    )

    predict_model_dir = fields.Str(
        load_default='/srv/tbbrdet_api/models/swin/coco/2023-12-07_130038',
        metadata={
            # 'enum': ls_folders(configs.MODEL_PATH, "best*.pth") +
            #         ls_folders(configs.REMOTE_MODEL_PATH, "best*.pth"),
            'description':
                'Model to be used for prediction. If only remote folders are '
                'available, the chosen one will be used and predictions saved '
                'remotely.\n\nCurrently existing "best" model paths are'
                '\n- locally:\n'
                f'{ls_folders(configs.MODEL_PATH, "best*.pth")}'
                '\n- remotely:\n'
                f'{ls_folders(configs.REMOTE_MODEL_PATH, "best*.pth")}\n'
        }
    )

    colour_channel = fields.Str(
        load_default="both",
        metadata={
            'enum': ["both", "RGB", "TIR"],
            'description': 'Image colour channels on which the predictions '
                           'will be visualized / saved to. '
                           'Choice of RGB, TIR or both side by side.'
        }
    )

    threshold = fields.Float(
        load_default=0.5,
        metadata={'description': 'Detection threshold.'}
    )

    device = fields.Bool(
        load_default=True,
        metadata={
            'enum': [True, False],
            'description': 'Computation device, default is GPU if GPU present.'
        }
    )

    # no_labels = fields.Bool(
    #     load_default=False,
    #     metadata={
    #         'enum': [True, False],
    #         'description': 'Visualize output only if this argument is '
    #                        'passed. Currently, this is not being used!'
    #     }
    # )

    accept = fields.Str(
        load_default='application/json',
        validate=validate.OneOf(['application/json']),
        # NOTE: can't handle 'image/png' at the moment
        metadata={
            'location': "headers",
            'description': "Define the type of output to get back. Returns "
                           "png file with detection results or a json with "
                           "the prediction. NOTE: Only json possible currently"
        }
    )


if __name__ == '__main__':
    pass
