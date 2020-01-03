RESULT_DIR := # result/BaselineModel-kwdlc-4e-large-coref-cz
CONFIG_FILE := # config/BaselineModel/kwdlc/4e/large-coref-cz.json
GPUS := # 0,1
TRAIN_ITER := 1  # number of train iteration with different random seed

CORPUS := kwdlc
TARGET := test
CSV_BASENAME := result_$(TARGET)_$(CORPUS).csv

SHELL = /bin/bash -exu
PYTHON := $(shell which python)

CHECKPOINTS := $(wildcard $(RESULT_DIR)/*/model_best.pth)
RESULT_FILES := $(patsubst $(RESULT_DIR)/%/model_best.pth,$(RESULT_DIR)/%/$(CSV_BASENAME),$(CHECKPOINTS))
SCORE_FILE := $(RESULT_DIR)/scores_$(TARGET)_$(CORPUS).csv

.PHONY: all train test
all: train test

ifdef CONFIG_FILE
	RESULT_DIR := result/$(subst /,-,$(patsubst config/%.json,%,$(CONFIG_FILE)))
endif

train:
	env n=$(TRAIN_ITER) gpu=$(GPUS) scripts/train.sh $(CONFIG_FILE)

test: $(SCORE_FILE)
	$(PYTHON) scripts/confidence_interval.py $^

$(SCORE_FILE): $(RESULT_FILES)
	cat <(ls $(RESULT_DIR)/*/$(CSV_BASENAME) | head -1 | xargs head -1) <(ls $(RESULT_DIR)/*/$(CSV_BASENAME) | xargs grep -h all_case) | tr -d ' ' | sed -r 's/[^,]+,//1' > $@ || rm -f $@

$(RESULT_FILES): $(RESULT_DIR)/%/$(CSV_BASENAME): $(RESULT_DIR)/%/model_best.pth
	$(PYTHON) src/test.py -r $< --target $(TARGET) -d $(GPUS)
