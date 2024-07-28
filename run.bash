#!/bin/bash
DATASET_PATH="/home/shuqi/dev/data/void/void_release/"
DEPTH_PREDICTOR="dpt_hybrid"
NSAMPLES=150
SML_MODEL_PATH="weights/sml_model.dpredictor.${DEPTH_PREDICTOR}.nsamples.${NSAMPLES}.ckpt"
python run.py -dp $DEPTH_PREDICTOR -ns $NSAMPLES -sm $SML_MODEL_PATH --save-output
