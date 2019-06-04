#!/usr/bin/env bash

set -exu

NN_BASED_ANAPHORA_RESOLUTION_DIR=/home/ueda/research/nn_based_anaphora_resolution
PYTHON2_COMMAND=/home/shibata/chainer-latest-24/bin/python
EVAL_CORPUS_BASEDIR=/share/tool/nn_based_anaphora_resolution/corpus/kwdlc
echo $#
OUTPUT_BASE_DIR=$1  # the first argument
if [[ $2 = "valid" ]]; then
    TARGET="dev"
elif [[ $2 = "test" ]]; then
    TARGET="test"
else
    echo "Error: The second argument must be 'valid' or 'test'."
    exit 1;
fi
OUT_KNP_DIR=${OUTPUT_BASE_DIR}/${TARGET}_out_knp
RESULT_FILE=${OUTPUT_BASE_DIR}/result.txt
RESULT_LOG_FILE=${OUTPUT_BASE_DIR}/result.log
RESULT_HTML_FILE=${OUTPUT_BASE_DIR}/result.html
RESULT_JSON_FILE=${OUTPUT_BASE_DIR}/result.json

# convert conll file to knp file
PYTHONPATH=${NN_BASED_ANAPHORA_RESOLUTION_DIR}/scripts python ${NN_BASED_ANAPHORA_RESOLUTION_DIR}/scripts/corpus/conll2knp.py --output_dir ${OUT_KNP_DIR} < ${OUTPUT_BASE_DIR}/$2_out.conll

# evaluate knp file
if [[ ${TARGET} = "test" ]]; then
    ${PYTHON2_COMMAND} ${NN_BASED_ANAPHORA_RESOLUTION_DIR}/scripts/scorer.py --knp_dir ${EVAL_CORPUS_BASEDIR}/knp_add_feature --test_id_file ${EVAL_CORPUS_BASEDIR}/test.files --system_dir ${OUT_KNP_DIR} --target ${TARGET} --inter_sentential --relax_evaluation --not_fix_case_analysis --relax_evaluation_multiple_argument --result_json ${RESULT_JSON_FILE} > ${RESULT_FILE} 2> ${RESULT_LOG_FILE}
else
    ${PYTHON2_COMMAND} ${NN_BASED_ANAPHORA_RESOLUTION_DIR}/scripts/scorer.py --knp_dir ${EVAL_CORPUS_BASEDIR}/knp_add_feature --dev_id_file ${EVAL_CORPUS_BASEDIR}/dev.files --system_dir ${OUT_KNP_DIR} --target ${TARGET} --inter_sentential --relax_evaluation --not_fix_case_analysis --relax_evaluation_multiple_argument --result_json ${RESULT_JSON_FILE} > /dev/null 2> ${RESULT_LOG_FILE}
fi

# convert result file to html format
if [[ ${TARGET} = "test" ]]; then
    ${PYTHON2_COMMAND} ${NN_BASED_ANAPHORA_RESOLUTION_DIR}/scripts/bert_result2html.py --result_file ${RESULT_FILE} > ${RESULT_HTML_FILE}
fi
