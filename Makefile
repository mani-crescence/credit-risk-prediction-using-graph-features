include .env
export QT_QPA_PLATFORM=offscreen
# Variables
PYTHON_INTERPRETER= python3
SRC_DIR = src/
PRE_DIR = -m src.data.

all: run
#LC
run_preprocess_lc:
	$(PYTHON_INTERPRETER) $(SRC_DIR)preprocessing.py $(DB_NAME) $(DB_PATH_LC) $(TARGET_NAME_LC) $(USELESS_ATTRIBUTES_LC)  $(TARGET_VALUES_LC) $(ATTRIBUTES_FOR_MANUAL_ENCODING_LC)  $(VALUES_FOR_MANUAL_ENCODING_LC)

run_sampling_lc:
	$(PYTHON_INTERPRETER) $(SRC_DIR)sampling_main.py $(DB_NAME) $(DB_PATH_LC) $(TARGET_NAME_LC)  $(USELESS_ATTRIBUTES_LC)  $(TARGET_VALUES_LC) $(ATTRIBUTES_FOR_MANUAL_ENCODING_LC)  $(VALUES_FOR_MANUAL_ENCODING_LC)

run_discretization_lc:
	$(PYTHON_INTERPRETER) $(SRC_DIR)discretization.py  $(TARGET_NAME_LC) $(DB_NAME)  $(T)

run_graph_modeling_lc:
	$(PYTHON_INTERPRETER) $(SRC_DIR)modeling.py $(DB_NAME) $(TARGET_NAME_LC) $(DISCRETIZATION_TYPE) $(GRAPH_TYPE) 		

run_compute_descriptors_lc:
	$(PYTHON_INTERPRETER) $(SRC_DIR)compute_descriptors.py $(TARGET_NAME_LC)  $(BD_NAME) $(ALPHA) $(GRAPH_TYPE) $(DISCRETIZATION_TYPE)

run_make_configurations_lc:
	$(PYTHON_INTERPRETER) $(SRC_DIR)build_configurations.py $(TARGET_NAME_LC) $(DB_NAME)  $(DISCRETIZATION_TYPE) $(GRAPH_TYPE)

run_make_predictions_lc:
	$(PYTHON_INTERPRETER) $(SRC_DIR)make_predictions_main.py $(TARGET_NAME_LC) $(DB_NAME)  "$(TRAIN_PATH)"  "$(TEST_PATH)"  $(DISCRETIZATION_TYPE) $(GRAPH_TYPE) $(CONFIG_PATH) $(ALPHA)

run_print_lc:
	$(PYTHON_INTERPRETER) $(SRC_DIR)print_result.py  $(DB_NAME)	$(DISCRETIZATION_TYPE) $(GRAPH_TYPE)

run_plot_lc:
	$(PYTHON_INTERPRETER) $(SRC_DIR)build_shap_plot.py $(TARGET_NAME_LC)  $(DB_NAME)  "$(TRAIN_DESCRIPTORS_PATHS)" "$(TEST_DESCRIPTORS_PATHS)"  $(MODEL) $(DISCRETIZATION_TYPE)

run_lc:
	$(PYTHON_INTERPRETER) $(SRC_DIR)single_launch.py  $(DB_NAME)


#GERMAN
run_preprocess_german:
	$(PYTHON_INTERPRETER) $(SRC_DIR)preprocessing.py $(DB_NAME) $(DB_PATH_GERMAN) $(TARGET_NAME_GERMAN) $(USELESS_ATTRIBUTES_GERMAN)  $(TARGET_VALUES_GERMAN) $(ATTIBUTES_FOR_MANUAL_ENCODING_GERMAN)  $(VALUES_FOR_MANUAL_ENCODING_GERMAN)

run_discretization_german:
	$(PYTHON_INTERPRETER) $(SRC_DIR)discretization.py $(TARGET_NAME_GERMAN) $(DB_NAME) $(T)

run_graph_modeling_german:
	$(PYTHON_INTERPRETER) $(SRC_DIR)modeling.py $(DB_NAME) $(TARGET_NAME_GERMAN) $(DISCRETIZATION_TYPE) $(GRAPH_TYPE)

run_compute_descriptors_german:
	$(PYTHON_INTERPRETER) $(SRC_DIR)compute_descriptors.py $(TARGET_NAME_GERMAN)  $(BD_NAME) $(ALPHA) $(GRAPH_TYPE) $(DISCRETIZATION_TYPE)

run_make_configurations_german:
	$(PYTHON_INTERPRETER) $(SRC_DIR)build_configurations.py $(TARGET_NAME_GERMAN) $(DB_NAME)  $(DISCRETIZATION_TYPE) $(GRAPH_TYPE)

run_make_predictions_german:
	$(PYTHON_INTERPRETER) $(SRC_DIR)make_predictions_main.py $(TARGET_NAME_GERMAN) $(DB_NAME)  "$(TRAIN_PATH)"  "$(TEST_PATH)"  $(DISCRETIZATION_TYPE) $(GRAPH_TYPE) $(CONFIG_PATH) $(ALPHA)

run_print_german:
	$(PYTHON_INTERPRETER) $(SRC_DIR)print_result.py  $(DB_NAME)	$(DISCRETIZATION_TYPE) $(GRAPH_TYPE)

run_plot_german:
	$(PYTHON_INTERPRETER) $(SRC_DIR)build_shap_plot.py $(TARGET_NAME_GERMAN)  $(DB_NAME)  "$(TRAIN_DESCRIPTORS_PATHS)" "$(TEST_DESCRIPTORS_PATHS)"  $(MODEL) $(DISCRETIZATION_TYPE)

run_summarize_german:
	$(PYTHON_INTERPRETER) $(SRC_DIR)summarize_shap_attributes_importance.py  $(DB_NAME)  $(DISCRETIZATION_TYPE)

run_german:
	$(PYTHON_INTERPRETER) $(SRC_DIR)single_launch.py  $(DB_NAME)


#LGD
run_preprocess_lgd:
	$(PYTHON_INTERPRETER) $(SRC_DIR)preprocessing.py $(DB_NAME) $(DB_PATH_LGD) $(TARGET_NAME_LGD) $(USELESS_ATTRIBUTES_LGD)  $(TARGET_VALUES_LGD) $(ATTIBUTES_FOR_MANUAL_ENCODING_LGD)  $(VALUES_FOR_MANUAL_ENCODING_LGD)

run_discretization_lgd:
	$(PYTHON_INTERPRETER) $(SRC_DIR)discretization.py $(TARGET_NAME_LGD) $(DB_NAME) $(T)

run_graph_modeling_lgd:
	$(PYTHON_INTERPRETER) $(SRC_DIR)modeling.py $(DB_NAME) $(TARGET_NAME_LGD) $(DISCRETIZATION_TYPE) $(GRAPH_TYPE)

run_compute_descriptors_lgd:
	$(PYTHON_INTERPRETER) $(SRC_DIR)compute_descriptors.py $(TARGET_NAME_LGD)  $(BD_NAME) $(ALPHA) $(GRAPH_TYPE) $(DISCRETIZATION_TYPE)

run_make_configurations_lgd:
	$(PYTHON_INTERPRETER) $(SRC_DIR)build_configurations.py $(TARGET_NAME_LGD) $(DB_NAME)  $(DISCRETIZATION_TYPE) $(GRAPH_TYPE)

run_make_predictions_lgd:
	$(PYTHON_INTERPRETER) $(SRC_DIR)make_predictions_main.py $(TARGET_NAME_LGD) $(DB_NAME)  "$(TRAIN_PATH)"  "$(TEST_PATH)"  $(DISCRETIZATION_TYPE) $(GRAPH_TYPE) $(CONFIG_PATH) $(ALPHA)

run_print_lgd:
	$(PYTHON_INTERPRETER) $(SRC_DIR)print_result.py  $(DB_NAME)	$(DISCRETIZATION_TYPE) $(GRAPH_TYPE)

run_plot_lgd:
	$(PYTHON_INTERPRETER) $(SRC_DIR)build_shap_plot.py $(TARGET_NAME_LGD)  $(DB_NAME)  "$(TRAIN_DESCRIPTORS_PATHS)" "$(TEST_DESCRIPTORS_PATHS)"  $(MODEL) $(DISCRETIZATION_TYPE)

run_lgd:
	$(PYTHON_INTERPRETER) $(SRC_DIR)single_launch.py  $(DB_NAME)

#JAPANESE
run_preprocess_japanese:
	$(PYTHON_INTERPRETER) $(SRC_DIR)preprocessing.py $(DB_NAME) $(DB_PATH_JAPANESE) $(TARGET_NAME_JAPANESE) $(USELESS_ATTRIBUTES_JAPANESE)  $(TARGET_VALUES_JAPANESE)

run_discretization_japanese:
	$(PYTHON_INTERPRETER) $(SRC_DIR)discretization.py $(TARGET_NAME_JAPANESE) $(DB_NAME) $(T)

run_graph_modeling_japanese:
	$(PYTHON_INTERPRETER) $(SRC_DIR)modeling.py $(DB_NAME) $(TARGET_NAME_JAPANESE) $(DISCRETIZATION_TYPE) $(GRAPH_TYPE)

run_compute_descriptors_japanese:
	$(PYTHON_INTERPRETER) $(SRC_DIR)compute_descriptors.py $(TARGET_NAME_JAPANESE)  $(BD_NAME) $(ALPHA) $(GRAPH_TYPE) $(DISCRETIZATION_TYPE)

run_make_configurations_japanese:
	$(PYTHON_INTERPRETER) $(SRC_DIR)build_configurations.py $(TARGET_NAME_JAPANESE) $(DB_NAME)  $(DISCRETIZATION_TYPE) $(GRAPH_TYPE)

run_make_predictions_japanese:
	$(PYTHON_INTERPRETER) $(SRC_DIR)make_predictions_main.py $(TARGET_NAME_JAPANESE) $(DB_NAME)  "$(TRAIN_PATH)"  "$(TEST_PATH)"  $(DISCRETIZATION_TYPE) $(GRAPH_TYPE) $(CONFIG_PATH) $(ALPHA)

run_print_japanese:
	$(PYTHON_INTERPRETER) $(SRC_DIR)print_result.py  $(DB_NAME)	$(DISCRETIZATION_TYPE) $(GRAPH_TYPE)

run_plot_japanese:
	$(PYTHON_INTERPRETER) $(SRC_DIR)build_shap_plot.py $(TARGET_NAME_JAPANESE)  $(DB_NAME)  "$(TRAIN_DESCRIPTORS_PATHS)" "$(TEST_DESCRIPTORS_PATHS)"  $(MODEL) $(DISCRETIZATION_TYPE)

run_summarize_japanese:
	$(PYTHON_INTERPRETER) $(SRC_DIR)summarize_shap_attributes_importance.py $(DB_NAME) $(DISCRETIZATION_TYPE)

run_japanese:
	$(PYTHON_INTERPRETER) $(SRC_DIR)single_launch.py  $(DB_NAME)

#HMEQ
run_preprocess_hmeq:
	$(PYTHON_INTERPRETER) $(SRC_DIR)preprocessing.py $(DB_NAME) $(DB_PATH_HMEQ) $(TARGET_NAME_HMEQ) $(USELESS_ATTRIBUTES_HMEQ)  $(TARGET_VALUES_HMEQ)

run_discretization_hmeq:
	$(PYTHON_INTERPRETER) $(SRC_DIR)discretization.py $(TARGET_NAME_HMEQ) $(DB_NAME) $(T)

run_graph_modeling_hmeq:
	$(PYTHON_INTERPRETER) $(SRC_DIR)modeling.py $(DB_NAME) $(TARGET_NAME_HMEQ) $(DISCRETIZATION_TYPE) $(GRAPH_TYPE)

run_compute_descriptors_hmeq:
	$(PYTHON_INTERPRETER) $(SRC_DIR)compute_descriptors.py $(TARGET_NAME_HMEQ)  $(BD_NAME) $(ALPHA) $(GRAPH_TYPE) $(DISCRETIZATION_TYPE)

run_make_configurations_hmeq:
	$(PYTHON_INTERPRETER) $(SRC_DIR)build_configurations.py $(TARGET_NAME_HMEQ) $(DB_NAME)  $(DISCRETIZATION_TYPE) $(GRAPH_TYPE)

run_make_predictions_hmeq:
	$(PYTHON_INTERPRETER) $(SRC_DIR)make_predictions_main.py $(TARGET_NAME_HMEQ) $(DB_NAME)  "$(TRAIN_PATH)"  "$(TEST_PATH)"  $(DISCRETIZATION_TYPE) $(GRAPH_TYPE) $(CONFIG_PATH) $(ALPHA)

run_print_hmeq:
	$(PYTHON_INTERPRETER) $(SRC_DIR)print_result.py  $(DB_NAME)	$(DISCRETIZATION_TYPE) $(GRAPH_TYPE)

run_plot_hmeq:
	$(PYTHON_INTERPRETER) $(SRC_DIR)build_shap_plot.py $(TARGET_NAME_HMEQ)  $(DB_NAME)  "$(TRAIN_DESCRIPTORS_PATHS)" "$(TEST_DESCRIPTORS_PATHS)"  $(MODEL) $(DISCRETIZATION_TYPE)

run_summarize_hmeq:
	$(PYTHON_INTERPRETER) $(SRC_DIR)summarize_shap_attributes_importance.py $(DB_NAME) $(DISCRETIZATION_TYPE)

run_hmeq:
	$(PYTHON_INTERPRETER) $(SRC_DIR)single_launch.py  $(DB_NAME)


#AUSTRALIAN
run_preprocess_australian:
	$(PYTHON_INTERPRETER) $(SRC_DIR)preprocessing.py $(DB_NAME) $(DB_PATH_AUSTRALIAN) $(TARGET_NAME_AUSTRALIAN) $(USELESS_ATTRIBUTES_AUSTRALIAN)  $(TARGET_VALUES_AUSTRALIAN)

run_discretization_australian:
	$(PYTHON_INTERPRETER) $(SRC_DIR)discretization.py $(TARGET_NAME_AUSTRALIAN) $(DB_NAME) $(T)

run_graph_modeling_australian:
	$(PYTHON_INTERPRETER) $(SRC_DIR)modeling.py $(DB_NAME) $(TARGET_NAME_AUSTRALIAN) $(DISCRETIZATION_TYPE) $(GRAPH_TYPE)

run_compute_descriptors_australian:
	$(PYTHON_INTERPRETER) $(SRC_DIR)compute_descriptors.py $(TARGET_NAME_AUSTRALIAN)  $(BD_NAME) $(ALPHA) $(GRAPH_TYPE) $(DISCRETIZATION_TYPE)

run_make_configurations_australian:
	$(PYTHON_INTERPRETER) $(SRC_DIR)build_configurations.py $(TARGET_NAME_AUSTRALIAN) $(DB_NAME)  $(DISCRETIZATION_TYPE) $(GRAPH_TYPE)

run_make_predictions_australian:
	$(PYTHON_INTERPRETER) $(SRC_DIR)make_predictions_main.py $(TARGET_NAME_AUSTRALIAN) $(DB_NAME)  "$(TRAIN_PATH)"  "$(TEST_PATH)"  $(DISCRETIZATION_TYPE) $(GRAPH_TYPE) $(CONFIG_PATH) $(ALPHA)

run_print_australian:
	$(PYTHON_INTERPRETER) $(SRC_DIR)print_result.py  $(DB_NAME)	$(DISCRETIZATION_TYPE) $(GRAPH_TYPE)

run_plot_australian:
	$(PYTHON_INTERPRETER) $(SRC_DIR)build_shap_plot.py $(TARGET_NAME_AUSTRALIAN)  $(DB_NAME)  "$(TRAIN_DESCRIPTORS_PATHS)" "$(TEST_DESCRIPTORS_PATHS)"  $(MODEL) $(DISCRETIZATION_TYPE)

run_summarize_australian:
	$(PYTHON_INTERPRETER) $(SRC_DIR)summarize_shap_attributes_importance.py $(DB_NAME) $(DISCRETIZATION_TYPE)

run_splitting_australian:
	$(PYTHON_INTERPRETER) $(PRE_DIR)splitting $(DB_NAME) $(DB_PATH_AUSTRALIAN)	$(TARGET_NAME_AUSTRALIAN) $(USELESS_ATTRIBUTES_AUSTRALIAN)  $(TARGET_VALUES_AUSTRALIAN)

run_engine_building_australian:
	$(PYTHON_INTERPRETER) $(PRE_DIR)build_preprocess_engine $(DB_NAME) $(TRAINSET_PATH_AUSTRALIAN)	 

run_australian:
	$(PYTHON_INTERPRETER) $(SRC_DIR)single_launch.py  $(DB_NAME)

#AER
run_sampling_aer:
	$(PYTHON_INTERPRETER) $(SRC_DIR)sampling_main.py $(DB_NAME) $(DB_PATH_AER) $(TARGET_NAME_AER) $(USELESS_ATTRIBUTES_AER) $(TARGET_NAME_AER)

run_preprocess_aer:
	$(PYTHON_INTERPRETER) $(SRC_DIR)preprocessing.py $(DB_NAME) $(DB_PATH_AER) $(TARGET_NAME_AER) $(USELESS_ATTRIBUTES_AER)  $(TARGET_VALUES_AER)

run_discretization_aer:
	$(PYTHON_INTERPRETER) $(SRC_DIR)discretization.py $(TARGET_NAME_AER) $(DB_NAME) $(T)

run_graph_modeling_aer:
	$(PYTHON_INTERPRETER) $(SRC_DIR)modeling.py $(DB_NAME) $(TARGET_NAME_AER) $(DISCRETIZATION_TYPE) $(GRAPH_TYPE)

run_compute_descriptors_aer:
	$(PYTHON_INTERPRETER) $(SRC_DIR)compute_descriptors.py $(TARGET_NAME_AER)  $(BD_NAME) $(ALPHA) $(GRAPH_TYPE) $(DISCRETIZATION_TYPE)

run_make_configurations_aer:
	$(PYTHON_INTERPRETER) $(SRC_DIR)build_configurations.py $(TARGET_NAME_AER) $(DB_NAME)  $(DISCRETIZATION_TYPE) $(GRAPH_TYPE)

run_make_predictions_aer:
	$(PYTHON_INTERPRETER) $(SRC_DIR)make_predictions_main.py $(TARGET_NAME_AER) $(DB_NAME)  "$(TRAIN_PATH)"  "$(TEST_PATH)"  $(DISCRETIZATION_TYPE) $(GRAPH_TYPE) $(CONFIG_PATH) $(ALPHA)

run_print_aer:
	$(PYTHON_INTERPRETER) $(SRC_DIR)print_result.py  $(DB_NAME)	$(DISCRETIZATION_TYPE) $(GRAPH_TYPE)

run_plot_aer:
	$(PYTHON_INTERPRETER) $(SRC_DIR)build_shap_plot.py $(TARGET_NAME_AER)  $(DB_NAME)  "$(TRAIN_DESCRIPTORS_PATHS)" "$(TEST_DESCRIPTORS_PATHS)"  $(MODEL) $(DISCRETIZATION_TYPE)

run_summarize_aer:
	$(PYTHON_INTERPRETER) $(SRC_DIR)summarize_shap_attributes_importance.py $(DB_NAME) $(DISCRETIZATION_TYPE)

run_aer:
	$(PYTHON_INTERPRETER) $(SRC_DIR)single_launch.py  $(DB_NAME)


#THOMAS
run_preprocess_thomas:
	$(PYTHON_INTERPRETER) $(SRC_DIR)preprocessing.py $(DB_NAME) $(DB_PATH_THOMAS) $(TARGET_NAME_THOMAS) $(USELESS_ATTRIBUTES_THOMAS)  $(TARGET_VALUES_THOMAS)

run_discretization_thomas:
	$(PYTHON_INTERPRETER) $(SRC_DIR)discretization.py $(TARGET_NAME_THOMAS) $(DB_NAME) $(T)

run_graph_modeling_thomas:
	$(PYTHON_INTERPRETER) $(SRC_DIR)modeling.py $(DB_NAME) $(TARGET_NAME_THOMAS) $(DISCRETIZATION_TYPE) $(GRAPH_TYPE)

run_compute_descriptors_thomas:
	$(PYTHON_INTERPRETER) $(SRC_DIR)compute_descriptors.py $(TARGET_NAME_THOMAS)  $(BD_NAME) $(ALPHA) $(GRAPH_TYPE) $(DISCRETIZATION_TYPE)

run_make_configurations_thomas:
	$(PYTHON_INTERPRETER) $(SRC_DIR)build_configurations.py $(TARGET_NAME_THOMAS) $(DB_NAME)  $(DISCRETIZATION_TYPE) $(GRAPH_TYPE)

run_make_predictions_thomas:
	$(PYTHON_INTERPRETER) $(SRC_DIR)make_predictions_main.py $(TARGET_NAME_THOMAS) $(DB_NAME)  "$(TRAIN_PATH)"  "$(TEST_PATH)"  $(DISCRETIZATION_TYPE) $(GRAPH_TYPE) $(CONFIG_PATH) $(ALPHA)

run_print_thomas:
	$(PYTHON_INTERPRETER) $(SRC_DIR)print_result.py  $(DB_NAME)	$(DISCRETIZATION_TYPE) $(GRAPH_TYPE)

run_plot_thomas:
	$(PYTHON_INTERPRETER) $(SRC_DIR)build_shap_plot.py $(TARGET_NAME_THOMAS)  $(DB_NAME)  "$(TRAIN_DESCRIPTORS_PATHS)" "$(TEST_DESCRIPTORS_PATHS)"  $(MODEL) $(DISCRETIZATION_TYPE)

run_thomas:
	$(PYTHON_INTERPRETER) $(SRC_DIR)single_launch.py  $(DB_NAME)


#KAGGLE-CREDIT-RISK
run_sampling_kaggle_credit_risk:
	$(PYTHON_INTERPRETER) $(SRC_DIR)sampling_main.py $(DB_NAME) $(DB_PATH_KAGGLE_CREDIT_RISK) $(TARGET_NAME_KAGGLE_CREDIT_RISK) $(USELESS_ATTRIBUTES_KAGGLE_CREDIT_RISK)  $(TARGET_NAME_KAGGLE_CREDIT_RISK)

run_preprocess_kaggle_credit_risk:
	$(PYTHON_INTERPRETER) $(SRC_DIR)preprocessing.py $(DB_NAME) $(DB_PATH_KAGGLE_CREDIT_RISK) $(TARGET_NAME_KAGGLE_CREDIT_RISK) $(USELESS_ATTRIBUTES_KAGGLE_CREDIT_RISK)  $(TARGET_VALUES_KAGGLE_CREDIT_RISK)

run_discretization_kaggle_credit_risk:
	$(PYTHON_INTERPRETER) $(SRC_DIR)discretization.py $(TARGET_NAME_KAGGLE_CREDIT_RISK) $(DB_NAME) $(T)

run_graph_modeling_kaggle_credit_risk:
	$(PYTHON_INTERPRETER) $(SRC_DIR)modeling.py $(DB_NAME) $(TARGET_NAME_KAGGLE_CREDIT_RISK) $(DISCRETIZATION_TYPE) $(GRAPH_TYPE)

run_compute_descriptors_kaggle_credit_risk:
	$(PYTHON_INTERPRETER) $(SRC_DIR)compute_descriptors.py $(TARGET_NAME_KAGGLE_CREDIT_RISK)  $(BD_NAME) $(ALPHA) $(GRAPH_TYPE) $(DISCRETIZATION_TYPE)

run_make_configurations_kaggle_credit_risk:
	$(PYTHON_INTERPRETER) $(SRC_DIR)build_configurations.py $(TARGET_NAME_KAGGLE_CREDIT_RISK) $(DB_NAME) $(DISCRETIZATION_TYPE) $(GRAPH_TYPE)

run_make_predictions_kaggle_credit_risk:
	$(PYTHON_INTERPRETER) $(SRC_DIR)make_predictions_main.py $(TARGET_NAME_KAGGLE_CREDIT_RISK) $(DB_NAME)  "$(TRAIN_PATH)"  "$(TEST_PATH)"  $(DISCRETIZATION_TYPE) $(GRAPH_TYPE) $(CONFIG_PATH) $(ALPHA)

run_print_kaggle_credit_risk:
	$(PYTHON_INTERPRETER) $(SRC_DIR)print_result.py  $(DB_NAME)	$(DISCRETIZATION_TYPE) $(GRAPH_TYPE)

run_plot_kaggle_credit_risk:
	$(PYTHON_INTERPRETER) $(SRC_DIR)build_shap_plot.py  $(TARGET_NAME_KAGGLE_CREDIT_RISK)  $(DB_NAME)  "$(TRAIN_DESCRIPTORS_PATHS)" "$(TEST_DESCRIPTORS_PATHS)"  $(MODEL) $(DISCRETIZATION_TYPE)

run_summarize_kaggle_credit_risk:
	$(PYTHON_INTERPRETER) $(SRC_DIR)summarize_shap_attributes_importance.py $(DB_NAME) $(DISCRETIZATION_TYPE)

run_kaggle_credit_risk:
	$(PYTHON_INTERPRETER) $(SRC_DIR)single_launch.py $(DB_NAME)

#MORTGAGE
run_sampling_mortgage:
	$(PYTHON_INTERPRETER) $(SRC_DIR)sampling_main.py $(DB_NAME) $(DB_PATH_MORTGAGE) $(TARGET_NAME_MORTGAGE) $(USELESS_ATTRIBUTES_MORTGAGE)   $(TARGET_VALUES_MORTGAGE)

run_preprocess_mortgage:
	$(PYTHON_INTERPRETER) $(SRC_DIR)preprocessing.py $(DB_NAME) $(DB_PATH_MORTGAGE) $(TARGET_NAME_MORTGAGE) $(USELESS_ATTRIBUTES_MORTGAGE)  $(TARGET_VALUES_MORTGAGE)

run_discretization_mortgage:
	$(PYTHON_INTERPRETER) $(SRC_DIR)discretization.py $(TARGET_NAME_MORTGAGE) $(DB_NAME) $(T)

run_graph_modeling_mortgage:
	$(PYTHON_INTERPRETER) $(SRC_DIR)modeling.py $(DB_NAME) $(TARGET_NAME_MORTGAGE) $(DISCRETIZATION_TYPE) $(GRAPH_TYPE)

run_compute_descriptors_mortgage:
	$(PYTHON_INTERPRETER) $(SRC_DIR)compute_descriptors.py $(TARGET_NAME_MORTGAGE)  $(BD_NAME) $(ALPHA) $(GRAPH_TYPE) $(DISCRETIZATION_TYPE)

run_make_configurations_mortgage:
	$(PYTHON_INTERPRETER) $(SRC_DIR)build_configurations.py $(TARGET_NAME_MORTGAGE) $(DB_NAME)  $(DISCRETIZATION_TYPE) $(GRAPH_TYPE)

run_make_predictions_mortgage:
	$(PYTHON_INTERPRETER) $(SRC_DIR)make_predictions_main.py $(TARGET_NAME_MORTGAGE) $(DB_NAME)  "$(TRAIN_PATH)"  "$(TEST_PATH)"  $(DISCRETIZATION_TYPE) $(GRAPH_TYPE) $(CONFIG_PATH) $(ALPHA)

run_print_mortgage:
	$(PYTHON_INTERPRETER) $(SRC_DIR)print_result.py  $(DB_NAME)	$(DISCRETIZATION_TYPE) $(GRAPH_TYPE)

run_plot_mortgage:
	$(PYTHON_INTERPRETER) $(SRC_DIR)build_shap_plot.py $(TARGET_NAME_MORTGAGE)  $(DB_NAME)  "$(TRAIN_DESCRIPTORS_PATHS)" "$(TEST_DESCRIPTORS_PATHS)"  $(MODEL) $(DISCRETIZATION_TYPE)

run_mortgage:
	$(PYTHON_INTERPRETER) $(SRC_DIR)single_launch.py  $(DB_NAME)

run_main_single_launch:
	$(PYTHON_INTERPRETER) $(SRC_DIR)main_single_launch.py $(DB_NAME)

run_main_print:
	$(PYTHON_INTERPRETER) $(SRC_DIR)print_result.py  $(DB_NAME)

run_general_launch:
	$(PYTHON_INTERPRETER) $(SRC_DIR)general_launch.py $(DB_NAME)

run_all:	run_main_single_launch	
#run_general_launch








































# #Bondora
# run_preprocess_bondora:
# 	 $(PYTHON_INTERPRETER) $(SRC_DIR)preprocessing.py $(DB_NAME) $(DB_PATH_BONDORA) $(TARGET_NAME_BONDORA)  $(USELESS_ATTRIBUTES_BONDORA)  $(COST_ATTRIBUTES_BONDORA) $(ATTIBUTES_FOR_MANUAL_ENCODING_BONDARA)  $(VALUES_FOR_MANUAL_ENCODING_BONDARA)

# run_sampling_bondora:
# 	 $(PYTHON_INTERPRETER) $(SRC_DIR)sampling_main.py $(DB_NAME) $(DB_PATH_BONDORA) $(TARGET_NAME_BONDORA)  $(USELESS_ATTRIBUTES_BONDORA)  $(COST_ATTRIBUTES_BONDORA) $(ATTIBUTES_FOR_MANUAL_ENCODING_BONDARA)  $(VALUES_FOR_MANUAL_ENCODING_BONDARA)

# run_discretization_bondora:
# 	 $(PYTHON_INTERPRETER) $(SRC_DIR)compute_descriptors_main.py $(TARGET_NAME_BONDORA) $(DB_NAME)  $(DISCRETIZATION_TYPE)

# run_graph_modeling_bondora:
# 	$(PYTHON_INTERPRETER) $(SRC_DIR)modeling.py $(DB_NAME) $(DISCRETIZATION_TYPE) $(GRAPH_TYPE)

# run_compute_descriptors_bondora:
# 	 $(PYTHON_INTERPRETER) $(SRC_DIR)compute_descriptors.py $(TARGET_NAME_BONDORA)  $(BD_NAME)  $(DISCRETIZATION_TYPE)  $(PAGERANK_TYPE) $(ALPHA) $(GRAPH_TYPE)

# run_make_configurations_bondora:
# 	$(PYTHON_INTERPRETER) $(SRC_DIR)build_configurations.py  $(TARGET_NAME_BONDORA) $(DB_NAME) $(DISCRETIZATION_TYPE) $(GRAPH_TYPE)

# run_make_predictions_bondora:
# 	$(PYTHON_INTERPRETER) $(SRC_DIR)make_predictions_main.py $(TARGET_NAME_BONDORA) $(COST_ATTRIBUTES_BONDORA) $(DB_NAME) $(TRAIN_PATH) $(TEST_PATH) $(DISCRETIZATION_TYPE) $(ALPHA)

# run_print_bondora:
# 	$(PYTHON_INTERPRETER) $(SRC_DIR)print_result.py  3 $(DB_NAME)  $(DISCRETIZATION_TYPE)

# run_plot_bondora:
# 	$(PYTHON_INTERPRETER) $(SRC_DIR)build_shap_plot.py  $(TARGET_NAME_BONDORA) $(DB_NAME)  $(DISCRETIZATION_TYPE) $(MODEL)

# run_bondora:
# 	$(PYTHON_INTERPRETER) $(SRC_DIR)launch.py $(DB_NAME)


# #Prosper
# run_preprocess_prosper:
# 	 $(PYTHON_INTERPRETER) $(SRC_DIR)preprocessing.py $(DB_NAME) $(DB_PATH_PROSPER) $(TARGET_NAME_PROSPER) $(USELESS_ATTRIBUTES_PROSPER)  $(COST_ATTRIBUTES_PROSPER) $(ATTIBUTES_FOR_MANUAL_ENCODING_PROSPER)  $(VALUES_FOR_MANUAL_ENCODING_PROSPER)

# run_sampling_prosper:
# 	 $(PYTHON_INTERPRETER) $(SRC_DIR)sampling_main.py $(DB_NAME) $(DB_PATH_PROSPER) $(TARGET_NAME_PROSPER)  $(USELESS_ATTRIBUTES_PROSPER)  $(COST_ATTRIBUTES_PROSPER) $(ATTIBUTES_FOR_MANUAL_ENCODING_PROSPER)  $(VALUES_FOR_MANUAL_ENCODING_PROSPER)

# run_discretization_prosper:
# 	 $(PYTHON_INTERPRETER) $(SRC_DIR)compute_descriptors_main.py $(TARGET_NAME_PROSPER) $(DB_NAME)  $(DISCRETIZATION_TYPE)

# run_graph_modeling_prosper:
# 	$(PYTHON_INTERPRETER) $(SRC_DIR)modeling.py $(DB_NAME) $(DISCRETIZATION_TYPE) $(GRAPH_TYPE)

# run_compute_descriptors_prosper:
# 	 $(PYTHON_INTERPRETER) $(SRC_DIR)compute_descriptors.py $(TARGET_NAME_PROSPER)  $(BD_NAME)  $(DISCRETIZATION_TYPE)  $(PAGERANK_TYPE) $(ALPHA) $(GRAPH_TYPE)

# run_make_configurations_prosper:
# 	$(PYTHON_INTERPRETER) $(SRC_DIR)build_configurations.py  $(TARGET_NAME_PROSPER) $(DB_NAME) $(DISCRETIZATION_TYPE) $(GRAPH_TYPE)

# run_make_predictions_prosper:
# 	$(PYTHON_INTERPRETER) $(SRC_DIR)make_predictions_main.py $(TARGET_NAME_PROSPER) $(COST_ATTRIBUTES_PROSPER) $(DB_NAME) $(TRAIN_PATH) $(TEST_PATH) $(DISCRETIZATION_TYPE) $(ALPHA)

# run_print_prosper:
# 	$(PYTHON_INTERPRETER) $(SRC_DIR)print_result.py  3  $(DB_NAME)  $(DISCRETIZATION_TYPE)

# run_plot_prosper:
# 	$(PYTHON_INTERPRETER) $(SRC_DIR)build_shap_plot.py  $(TARGET_NAME_PROSPER) $(DB_NAME)  $(DISCRETIZATION_TYPE)  $(MODEL)

# run_prosper:
# 	$(PYTHON_INTERPRETER) $(SRC_DIR)launch.py $(DB_NAME)

# #ECAI
# run_preprocess_ecai:
# 	 $(PYTHON_INTERPRETER) $(SRC_DIR)preprocessing.py  $(DB_NAME) $(DB_PATH_ECAI) $(TARGET_NAME_ECAI)  $(USELESS_ATTRIBUTES_ECAI)  $(COST_ATTRIBUTES_ECAI) $(ATTIBUTES_FOR_MANUAL_ENCODING_ECAI)  $(VALUES_FOR_MANUAL_ENCODING_ECAI)

# run_discretization_ecai:
# 	 $(PYTHON_INTERPRETER) $(SRC_DIR)compute_descriptors_main.py $(TARGET_NAME_ECAI) $(DB_NAME)  $(DISCRETIZATION_TYPE)

# run_graph_modeling_ecai:
# 	$(PYTHON_INTERPRETER) $(SRC_DIR)modeling.py $(DB_NAME) $(DISCRETIZATION_TYPE) $(GRAPH_TYPE)

# run_compute_descriptors_ecai:
# 	 $(PYTHON_INTERPRETER) $(SRC_DIR)compute_descriptors.py $(TARGET_NAME_ECAI)  $(BD_NAME)  $(DISCRETIZATION_TYPE) $(PAGERANK_TYPE) $(ALPHA) $(GRAPH_TYPE)

# run_make_configurations_ecai:
# 	$(PYTHON_INTERPRETER) $(SRC_DIR)build_configurations.py $(TARGET_NAME_ECAI) $(DB_NAME) $(PROCESS_TYPE)

# run_make_predictions_ecai:
# 	$(PYTHON_INTERPRETER) $(SRC_DIR)make_predictions_main.py $(TARGET_NAME_ECAI) $(COST_ATTRIBUTES_ECAI) $(DB_NAME) $(TRAIN_PATH) $(TEST_PATH) $(DISCRETIZATION_TYPE) $(ALPHA)

# run_print_ecai:
# 	$(PYTHON_INTERPRETER) $(SRC_DIR)print_result.py  2  $(DB_NAME)  $(DISCRETIZATION_TYPE)

# run_plot_ecai:
# 	$(PYTHON_INTERPRETER) $(SRC_DIR)build_shap_plot.py  $(TARGET_NAME_ECAI) $(DB_NAME)  $(DISCRETIZATION_TYPE) $(MODEL)

# run_ecai:
# 	$(PYTHON_INTERPRETER) $(SRC_DIR)launch.py $(DB_NAME)
