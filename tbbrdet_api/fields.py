#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Selectable options to train and test a model.
A platform user can enter information in fields that equate to the below options.

Based on: K Alibabaei's fasterrcnn_pytorch_api.git
https://git.scc.kit.edu/m-team/ai/fasterrcnn_pytorch_api/-/blob/master/fasterrcnn_pytorch_api/fields.py
"""
from webargs import fields, validate
from marshmallow import Schema, fields, validates_schema, ValidationError
from tbbrdet_api import configs
from tbbrdet_api.misc import (
    ls_remote, ls_local, get_model_dirs, get_pretrain_ckpt_paths
)

# --------------------------------------

REMOTE_PTHS = ls_remote()
LOCAL_PTHS = ls_local()


class TrainArgsSchema(Schema):
    """
    Class of all selectable options to train the model
    """

    class Meta:
        ordered = True

    # note: currently only Mask-RCNN ResNet 50 FPN - scratch and pretrained.
    #       "Weights" below defines pretrained or not. Model choice here is therefore unnecessary.
    model = fields.Str(
        required=True,
        metadata={
            'enum': configs.BACKBONES,
            "description": "Model name."
        }
    )

    ckp_pretrain_pth = fields.Str(
        required=False,
        load_default=None,
        metadata={
            'enum': get_pretrain_ckpt_paths(LOCAL_PTHS) + get_pretrain_ckpt_paths(REMOTE_PTHS),
            'description': 'If you want to train a new model with pretrained weights, choose a '
                           'ckp_pretrain_pth file from the provided options. If only remote paths '
                           'are available, the chosen one will be downloaded. '
                           'You cannot combine this with resuming training.'
        }
    )

    ckp_resume_dir = fields.Str(
        required=False,
        load_default=None,
        metadata={
            'enum': get_model_dirs(LOCAL_PTHS) + get_model_dirs(REMOTE_PTHS),
            'description': 'If you want to resume training a model, choose a folder file from '
                           'the provided options. If only remote paths are available, '
                           'the chosen associated folder will be downloaded. You cannot combine '
                           'this with training a new model with pretrained weights.'
        }
    )

    # # note: data config isn't necessary here, we have the mmdet configs
    # data_config = fields.Str(
    #     required=False,
    #     metadata={
    #         'description': 'Path to the data config file.'
    #         }
    #     )

    device = fields.Bool(
        required=False,
        load_default=True,
        metadata={
            'enum': [True, False],
            'description': "Computation/training device. The default is a GPU."
                           "Training won't work without a GPU"
        }
    )

    epochs = fields.Int(
        required=False,
        load_default=4,
        metadata={'description': 'Number of epochs to train.'}
    )

    workers = fields.Int(
        required=False,
        load_default=2,
        metadata={'description': 'Number of workers for data processing/transforms/augmentations.'}
    )

    batch = fields.Int(
        required=False,
        load_default=1,
        metadata={'description': 'Batch size to load the data.'}
    )

    lr = fields.Float(
        required=False,
        load_default=0.0001,
        metadata={'description': 'Learning rate.'}
    )

    seed = fields.Int(
        required=False,
        load_default=0,
        metadata={'description': 'Global seed number for training.'}
    )

    @validates_schema
    def validate_required_fields(self, data):
        if 'ckp_resume_dir' in data and 'ckp_pretrain_pth' in data:
            raise ValidationError('Only either a model ckp_pretrain_pth path OR a checkpoint path '
                                  'can be used at once.')
    # todo: Add ValidationError for training without a GPU (will otherwise fail in scripts.train)


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
        required=True,
        metadata={
            'enum': get_model_dirs(LOCAL_PTHS) + get_model_dirs(REMOTE_PTHS),
            'description': 'Model to be used for prediction. If only remote folders '
                           'are available, the chosen one will be downloaded.'
        }
    )

    colour_channel = fields.Str(
        required=True,
        metadata={
            'enum': ["RGB", "Thermal"],
            'description': 'Image colour channels on which the predictions will be visualized / '
                           'saved to. Choice of RGB or Thermal.'
        }
    )

    threshold = fields.Float(
        required=False,
        load_default=0.5,
        metadata={'description': 'Detection threshold.'}
    )

    device = fields.Bool(
        required=False,
        load_default=True,
        metadata={
            'enum': [True, False],
            'description': 'Computation device, default is GPU if GPU present.'
        }
    )

    no_labels = fields.Bool(
        required=False,
        load_default=False,
        metadata={
            'enum': [True, False],
            'description': 'Visualize output only if this argument is passed.'
        }
    )

    accept = fields.Str(
        load_default="application/pdf",
        validate=validate.OneOf(['image/png', 'application/json']),
        metadata={
            'location': "headers",
            'description': "Define the type of output to get back. Returns png file with "
                           "detection results r a json with the prediction."
        }
    )


if __name__ == '__main__':
    pass
