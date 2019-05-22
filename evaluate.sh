#!/usr/bin/env bash

set -exu

NN_BASED_ANAPHORA_RESOLUTION_DIR=/home/ueda/research/nn_based_anaphora_resolution
PYTHON2_COMMAND=/home/shibata/chainer-latest-24/bin/python
EVAL_CORPUS_BASEDIR=/share/tool/nn_based_anaphora_resolution/corpus/kwdlc

OUTPUT_BASE_DIR=$1  # the first argument
TARGET=$2  # the second argument
OUT_KNP_DIR=${OUTPUT_BASE_DIR}/${TARGET}_out_knp
RESULT_FILE=${OUTPUT_BASE_DIR}/result.txt

# convert conll file to knp file
PYTHONPATH=${NN_BASED_ANAPHORA_RESOLUTION_DIR}/scripts python ${NN_BASED_ANAPHORA_RESOLUTION_DIR}/scripts/corpus/conll2knp.py --output_dir ${OUT_KNP_DIR} < ${OUTPUT_BASE_DIR}/${TARGET}_out.conll
# evaluate knp file
${PYTHON2_COMMAND} ${NN_BASED_ANAPHORA_RESOLUTION_DIR}/scripts/scorer.py --knp_dir ${EVAL_CORPUS_BASEDIR}/knp_add_feature --dev_id_file ${EVAL_CORPUS_BASEDIR}/dev.files --test_id_file ${EVAL_CORPUS_BASEDIR}/test.files --system_dir ${OUT_KNP_DIR} --target ${TARGET} --inter_sentential --relax_evaluation --not_fix_case_analysis --relax_evaluation_multiple_argument > ${RESULT_FILE} 2> ${RESULT_FILE}.log
# convert result file to html format
$(PYTHON2_COMMAND) $(NN_BASED_ANAPHORA_RESOLUTION_DIR)/scripts/bert_result2html.py --result_file $< > $@
