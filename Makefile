include .env
export QT_QPA_PLATFORM=offscreen

# Variables
PYTHON_INTERPRETER= python3
SRC_DIR = -m src.
PRE_DIR = -m src.data_processing.

all: run

run_all_dbs_with_normalization:
	$(PYTHON_INTERPRETER) $(SRC_DIR)launchers.with_normalization.launch_all_databases $(DB_NAME)

run_all_dbs_without_normalization:
	$(PYTHON_INTERPRETER) $(SRC_DIR)launchers.without_normalization.launch_all_databases $(DB_NAME)	

run_summarize_data_without_normalization:
	$(PYTHON_INTERPRETER) $(SRC_DIR)launchers.without_normalization.launch_data_summarization $(DB_NAME) 

run_summarize_data_with_normalization:
	$(PYTHON_INTERPRETER) $(SRC_DIR)launchers.with_normalization.launch_data_summarization $(DB_NAME) 

run_all: run_all_dbs_without_normalization	run_all_dbs_with_normalization

run_summarization:  run_summarize_data_with_normalization # run_summarize_data_without_normalization

run_graph: 
	$(PYTHON_INTERPRETER) $(SRC_DIR)main_graph_launch $(DB_NAME)




#BONDORA

########################################################### DATA PREPROCESSING COMMANDS #######################################################################################

run_splitting_bondora:
	$(PYTHON_INTERPRETER) $(PRE_DIR)splitting $(DB_NAME) $(DB_PATH_BONDORA)	$(TARGET_NAME_BONDORA) $(USELESS_ATTRIBUTES_BONDORA)  $(TARGET_VALUES_BONDORA)

run_engine_building_pre_bondora:
	$(PYTHON_INTERPRETER) $(PRE_DIR)build_preprocess_engine $(DB_NAME) $(TRAINSET_PATH_BONDORA) $(TARGET_NAME_BONDORA)

run_engine_building_unsupervised_discretization_bondora:
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.engine.build_unsupervised_discretization_engine $(DB_NAME) $(TARGET_NAME_BONDORA) "$(_PATH)"  "$(_DIR)" 

run_engine_building_supervised_discretization_bondora:
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.engine.build_supervised_discretization_engine $(DB_NAME) $(TARGET_NAME_BONDORA) "$(_PATH)" "$(_DIR)" 

run_preprocess_train_bondora:
	$(PYTHON_INTERPRETER) $(PRE_DIR)preprocessing $(DB_NAME) $(TRAINSET_PATH_BONDORA) $(TARGET_NAME_BONDORA) $(USELESS_ATTRIBUTES_BONDORA)  $(TRAIN_LABEL)

run_preprocess_test_bondora:
	$(PYTHON_INTERPRETER) $(PRE_DIR)preprocessing $(DB_NAME) $(TESTSET_PATH_BONDORA) $(TARGET_NAME_BONDORA) $(USELESS_ATTRIBUTES_BONDORA)  $(TEST_LABEL)

run_preprocess_bondora:
	$(PYTHON_INTERPRETER) $(PRE_DIR)preprocessing_of_state_of_art $(DB_NAME) $(DB_PATH_BONDORA) $(TARGET_NAME_BONDORA) $(USELESS_ATTRIBUTES_BONDORA)  

run_supervised_discretization_bondora:
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.supervised_discretization_process  $(DB_NAME)  $(TARGET_NAME_BONDORA)  "$(_PATH)"  $(NORMALIZATION_LABEL)  $(DATA_LABEL) 

run_unsupervised_discretization_bondora:
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.unsupervised_discretization_process  $(DB_NAME)  $(TARGET_NAME_BONDORA)  "$(_PATH)"  $(NORMALIZATION_LABEL)  $(DATA_LABEL) 

	
########################################################################### GRAPH MANAGEMENT COMMANDS #########################################################################

# BIP GRAPH	
run_graph_modeling_bip_bondora:
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.graph.bip.build_graph $(DB_NAME) $(TARGET_NAME_BONDORA)  $(GRAPH_TYPE) $(DISCRETIZATION_TYPE) "$(TRAIN_PATH)" "$(TEST_PATH)"  "$(_DIR)"

run_compute_descriptors_bip_bondora: 
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.graph.bip.compute_descriptors $(TARGET_NAME_BONDORA)  $(BD_NAME) $(GRAPH_TYPE) $(ALPHA)  $(DISCRETIZATION_TYPE) "$(TRAIN_PATH)" "$(TEST_PATH)"  "$(_DIR)"  "$(GRAPH_DIR)"


# MOD GRAPH	 
run_graph_modeling_mod_bondora:
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.graph.mod.build_graph $(DB_NAME) $(TARGET_NAME_BONDORA)  $(GRAPH_TYPE) $(DISCRETIZATION_TYPE) "$(TRAIN_PATH)" "$(TEST_PATH)"  "$(_DIR)"

run_compute_descriptors_mod_bondora: 
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.graph.mod.compute_descriptors $(TARGET_NAME_BONDORA)  $(BD_NAME) $(GRAPH_TYPE) $(ALPHA)  $(DISCRETIZATION_TYPE) "$(TRAIN_PATH)" "$(TEST_PATH)"  "$(_DIR)"  "$(GRAPH_DIR)"



# COMPLETE GRAPHS MANAGEMENT
run_graph_modeling_complete_bondora:
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.graph.complete.build_graph $(DB_NAME)  "$(_DIR)"

run_create_edges_of_train_bondora: 	
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.graph.complete.create_edges_of_train $(DB_NAME) $(START) $(END) "$(_PATH)" "$(_DIR)"

run_create_edges_of_test_bondora: 	
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.graph.complete.create_edges_of_test $(DB_NAME) $(START) $(END) "$(_PATH)" "$(_DIR)"

run_relate_edges_of_train_bondora: 	
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.graph.complete.relate_edges_of_train $(DB_NAME) $(START1) $(END1) $(START2) $(END2) "$(_PATH)"  "$(_DIR)"

run_relate_edges_of_test_bondora: 	
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.graph.complete.relate_edges_of_test $(DB_NAME) $(START1) $(END1) $(START2) $(END2) "$(_PATH)"  "$(_DIR)"

run_relate_edges_of_both_bondora: 	
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.graph.complete.relate_edges_of_both $(DB_NAME) $(START1) $(END1) $(START2) $(END2) "$(TRAIN_PATH)" "$(TEST_PATH)" "$(_DIR)"

run_compute_descriptors_liu_v1_bondora: 
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.graph.liu.compute_descriptors_v1 $(TARGET_NAME_BONDORA)  $(BD_NAME) $(GRAPH_TYPE) "$(TRAIN_PATH)"  "$(TEST_PATH)"  "$(_DIR)"  "$(GRAPH_DIR)"

run_compute_descriptors_liu_v2_bondora: 
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.graph.liu.compute_descriptors_v2 $(TARGET_NAME_BONDORA)  $(BD_NAME) $(GRAPH_TYPE) "$(TRAIN_PATH)"  "$(TEST_PATH)"  "$(_DIR)"  "$(GRAPH_DIR)"

run_compute_descriptors_gui_bondora: 
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.graph.gui.compute_descriptors $(TARGET_NAME_BONDORA)  $(BD_NAME) $(GRAPH_TYPE) "$(TRAIN_PATH)"  "$(TEST_PATH)"  "$(_DIR)"  "$(GRAPH_DIR)" 

run_graph_modeling_gui_bondora:
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.graph.gui.build_graph $(DB_NAME) $(TARGET_NAME_BONDORA)  $(GRAPH_TYPE) $(DISCRETIZATION_TYPE) "$(TRAIN_PATH)" "$(TEST_PATH)"  "$(_DIR)"

run_graph_modeling_liu_bondora:
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.graph.liu.build_graph $(DB_NAME) $(TARGET_NAME_BONDORA)  $(GRAPH_TYPE) $(DISCRETIZATION_TYPE) "$(TRAIN_PATH)" "$(TEST_PATH)"  "$(_DIR)"	

# GLO GRAPH
run_compute_descriptors_glo_bondora:
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.graph.complete.glo.compute_descriptors  $(TARGET_NAME_BONDORA)  $(BD_NAME) $(GRAPH_TYPE) $(ALPHA)	

run_graph_glo_bondora:
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.graph.complete.glo.launch  $(DB_NAME)	

# LOAN GRAPH	
run_graph_modeling_loan_bondora:
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.graph.loan.build_graph $(DB_NAME) $(TARGET_NAME_BONDORA) $(GRAPH_TYPE) "$(TRAIN_PATH)"  "$(TEST_PATH)"  "$(_DIR)"

run_compute_descriptors_loan_bondora: 
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.graph.loan.compute_descriptors $(TARGET_NAME_BONDORA)  $(BD_NAME)  $(GRAPH_TYPE)  $(ALPHA)  "$(TRAIN_PATH)"  "$(TEST_PATH)"  "$(GRAPH_DIR)"  "$(_DIR)"

###############################################################################################################################################################

run_make_configurations_bondora:
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.build_configurations $(TARGET_NAME_BONDORA) $(DB_NAME) $(SAVE_DIR) $(DISCRETIZATION_TYPE) $(GRAPH_TYPE) $(CLASSIC_TRAIN_PATH) $(NEW_DESCRIPTOR_TRAIN_PATH) 

run_make_configurations_with_stepwise_bondora:
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.build_configurations_with_stepwise $(TARGET_NAME_BONDORA) $(DB_NAME) $(SAVE_DIR) $(DISCRETIZATION_TYPE) $(GRAPH_TYPE) $(CLASSIC_TRAIN_PATH) $(NEW_DESCRIPTOR_TRAIN_PATH) 

run_select_features_bondora:
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.select_features_with_pvalue $(TARGET_NAME_BONDORA) $(DB_NAME) "$(TRAIN_PATH)" "$(SAVE_DIR)"
	
run_make_predictions_bondora:
	$(PYTHON_INTERPRETER) $(SRC_DIR)models.make_predictions_main $(TARGET_NAME_BONDORA) $(DB_NAME)  "$(TRAIN_PATH)"  "$(TEST_PATH)"  $(DISCRETIZATION_TYPE) $(GRAPH_TYPE) "$(CONFIG_PATH)" "$(SAVE_DIR)"  "$(CLASSIC_TRAIN_PATH)"  "$(CLASSIC_TEST_PATH)" "$(CLASSIC_CONFIG_PATH)" $(ALPHA) 

run_print_bondora:
	$(PYTHON_INTERPRETER) $(SRC_DIR)visualization.print_result  $(DB_NAME)	"$(_DIR)"  $(DISCRETIZATION_TYPE) $(GRAPH_TYPE)

run_plot_bondora:
	$(PYTHON_INTERPRETER) $(SRC_DIR)build_shap_plot.py $(TARGET_NAME_BONDORA)  $(DB_NAME)  "$(TRAIN_DESCRIPTORS_PATHS)" "$(TEST_DESCRIPTORS_PATHS)"  $(MODEL) $(DISCRETIZATION_TYPE)

run_summarize_bondora:
	$(PYTHON_INTERPRETER) $(SRC_DIR)summarize_shap_attributes_importance.py $(DB_NAME) $(DISCRETIZATION_TYPE)

run_without_normalization_bondora:
	$(PYTHON_INTERPRETER) $(SRC_DIR)launchers.without_normalization.launch_per_database $(DB_NAME)

run_with_normalization_bondora:
	$(PYTHON_INTERPRETER) $(SRC_DIR)launchers.with_normalization.launch_per_database  $(DB_NAME)




#PROSPER

########################################################### DATA PREPROCESSING COMMANDS #######################################################################################

run_splitting_prosper:
	$(PYTHON_INTERPRETER) $(PRE_DIR)splitting $(DB_NAME) $(DB_PATH_PROSPER)	$(TARGET_NAME_PROSPER) $(USELESS_ATTRIBUTES_PROSPER)  $(TARGET_VALUES_PROSPER)

run_engine_building_pre_prosper:
	$(PYTHON_INTERPRETER) $(PRE_DIR)build_preprocess_engine $(DB_NAME) $(TRAINSET_PATH_PROSPER) $(TARGET_NAME_PROSPER)

run_engine_building_unsupervised_discretization_prosper:
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.engine.build_unsupervised_discretization_engine $(DB_NAME) $(TARGET_NAME_PROSPER) "$(_PATH)"  "$(_DIR)" 

run_engine_building_supervised_discretization_prosper:
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.engine.build_supervised_discretization_engine $(DB_NAME) $(TARGET_NAME_PROSPER) "$(_PATH)" "$(_DIR)" 

run_preprocess_train_prosper:
	$(PYTHON_INTERPRETER) $(PRE_DIR)preprocessing $(DB_NAME) $(TRAINSET_PATH_PROSPER) $(TARGET_NAME_PROSPER) $(USELESS_ATTRIBUTES_PROSPER)  $(TRAIN_LABEL)

run_preprocess_test_prosper:
	$(PYTHON_INTERPRETER) $(PRE_DIR)preprocessing $(DB_NAME) $(TESTSET_PATH_PROSPER) $(TARGET_NAME_PROSPER) $(USELESS_ATTRIBUTES_PROSPER)  $(TEST_LABEL)

run_preprocess_prosper:
	$(PYTHON_INTERPRETER) $(PRE_DIR)preprocessing_of_state_of_art $(DB_NAME) $(DB_PATH_PROSPER) $(TARGET_NAME_PROSPER) $(USELESS_ATTRIBUTES_PROSPER)  

run_supervised_discretization_prosper:
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.supervised_discretization_process  $(DB_NAME)  $(TARGET_NAME_PROSPER)  "$(_PATH)"  $(NORMALIZATION_LABEL)  $(DATA_LABEL) 

run_unsupervised_discretization_prosper:
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.unsupervised_discretization_process  $(DB_NAME)  $(TARGET_NAME_PROSPER)  "$(_PATH)"  $(NORMALIZATION_LABEL)  $(DATA_LABEL) 

	
########################################################################### GRAPH MANAGEMENT COMMANDS #########################################################################

# BIP GRAPH	
run_graph_modeling_bip_prosper:
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.graph.bip.build_graph $(DB_NAME) $(TARGET_NAME_PROSPER)  $(GRAPH_TYPE) $(DISCRETIZATION_TYPE) "$(TRAIN_PATH)" "$(TEST_PATH)"  "$(_DIR)"

run_compute_descriptors_bip_prosper: 
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.graph.bip.compute_descriptors $(TARGET_NAME_PROSPER)  $(BD_NAME) $(GRAPH_TYPE) $(ALPHA)  $(DISCRETIZATION_TYPE) "$(TRAIN_PATH)" "$(TEST_PATH)"  "$(_DIR)"  "$(GRAPH_DIR)"


# MOD GRAPH	 
run_graph_modeling_mod_prosper:
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.graph.mod.build_graph $(DB_NAME) $(TARGET_NAME_PROSPER)  $(GRAPH_TYPE) $(DISCRETIZATION_TYPE) "$(TRAIN_PATH)" "$(TEST_PATH)"  "$(_DIR)"

run_compute_descriptors_mod_prosper: 
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.graph.mod.compute_descriptors $(TARGET_NAME_PROSPER)  $(BD_NAME) $(GRAPH_TYPE) $(ALPHA)  $(DISCRETIZATION_TYPE) "$(TRAIN_PATH)" "$(TEST_PATH)"  "$(_DIR)"  "$(GRAPH_DIR)"



# COMPLETE GRAPHS MANAGEMENT
run_graph_modeling_complete_prosper:
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.graph.complete.build_graph $(DB_NAME)  "$(_DIR)"

run_create_edges_of_train_prosper: 	
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.graph.complete.create_edges_of_train $(DB_NAME) $(START) $(END) "$(_PATH)" "$(_DIR)"

run_create_edges_of_test_prosper: 	
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.graph.complete.create_edges_of_test $(DB_NAME) $(START) $(END) "$(_PATH)" "$(_DIR)"

run_relate_edges_of_train_prosper: 	
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.graph.complete.relate_edges_of_train $(DB_NAME) $(START1) $(END1) $(START2) $(END2) "$(_PATH)"  "$(_DIR)"

run_relate_edges_of_test_prosper: 	
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.graph.complete.relate_edges_of_test $(DB_NAME) $(START1) $(END1) $(START2) $(END2) "$(_PATH)"  "$(_DIR)"

run_relate_edges_of_both_prosper: 	
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.graph.complete.relate_edges_of_both $(DB_NAME) $(START1) $(END1) $(START2) $(END2) "$(TRAIN_PATH)" "$(TEST_PATH)" "$(_DIR)"

run_compute_descriptors_liu_v1_prosper: 
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.graph.liu.compute_descriptors_v1 $(TARGET_NAME_PROSPER)  $(BD_NAME) $(GRAPH_TYPE) "$(TRAIN_PATH)"  "$(TEST_PATH)"  "$(_DIR)"  "$(GRAPH_DIR)"

run_compute_descriptors_liu_v2_prosper: 
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.graph.liu.compute_descriptors_v2 $(TARGET_NAME_PROSPER)  $(BD_NAME) $(GRAPH_TYPE) "$(TRAIN_PATH)"  "$(TEST_PATH)"  "$(_DIR)"  "$(GRAPH_DIR)"

run_compute_descriptors_gui_prosper: 
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.graph.gui.compute_descriptors $(TARGET_NAME_PROSPER)  $(BD_NAME) $(GRAPH_TYPE) "$(TRAIN_PATH)"  "$(TEST_PATH)"  "$(_DIR)"  "$(GRAPH_DIR)" 

run_graph_modeling_gui_prosper:
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.graph.gui.build_graph $(DB_NAME) $(TARGET_NAME_PROSPER)  $(GRAPH_TYPE) $(DISCRETIZATION_TYPE) "$(TRAIN_PATH)" "$(TEST_PATH)"  "$(_DIR)"

run_graph_modeling_liu_prosper:
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.graph.liu.build_graph $(DB_NAME) $(TARGET_NAME_PROSPER)  $(GRAPH_TYPE) $(DISCRETIZATION_TYPE) "$(TRAIN_PATH)" "$(TEST_PATH)"  "$(_DIR)"	

# GLO GRAPH
run_compute_descriptors_glo_prosper:
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.graph.complete.glo.compute_descriptors  $(TARGET_NAME_PROSPER)  $(BD_NAME) $(GRAPH_TYPE) $(ALPHA)	

run_graph_glo_prosper:
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.graph.complete.glo.launch  $(DB_NAME)	

# LOAN GRAPH	
run_graph_modeling_loan_prosper:
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.graph.loan.build_graph $(DB_NAME) $(TARGET_NAME_PROSPER) $(GRAPH_TYPE) "$(TRAIN_PATH)"  "$(TEST_PATH)"  "$(_DIR)"

run_compute_descriptors_loan_prosper: 
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.graph.loan.compute_descriptors $(TARGET_NAME_PROSPER)  $(BD_NAME)  $(GRAPH_TYPE)  $(ALPHA)  "$(TRAIN_PATH)"  "$(TEST_PATH)"  "$(GRAPH_DIR)"  "$(_DIR)"

###############################################################################################################################################################

run_make_configurations_prosper:
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.build_configurations $(TARGET_NAME_PROSPER) $(DB_NAME) $(SAVE_DIR) $(DISCRETIZATION_TYPE) $(GRAPH_TYPE) $(CLASSIC_TRAIN_PATH) $(NEW_DESCRIPTOR_TRAIN_PATH) 

run_make_configurations_with_stepwise_prosper:
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.build_configurations_with_stepwise $(TARGET_NAME_PROSPER) $(DB_NAME) $(SAVE_DIR) $(DISCRETIZATION_TYPE) $(GRAPH_TYPE) $(CLASSIC_TRAIN_PATH) $(NEW_DESCRIPTOR_TRAIN_PATH) 

run_select_features_prosper:
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.select_features_with_pvalue $(TARGET_NAME_PROSPER) $(DB_NAME) "$(TRAIN_PATH)" "$(SAVE_DIR)"
	
run_make_predictions_prosper:
	$(PYTHON_INTERPRETER) $(SRC_DIR)models.make_predictions_main $(TARGET_NAME_PROSPER) $(DB_NAME)  "$(TRAIN_PATH)"  "$(TEST_PATH)"  $(DISCRETIZATION_TYPE) $(GRAPH_TYPE) "$(CONFIG_PATH)" "$(SAVE_DIR)"  "$(CLASSIC_TRAIN_PATH)"  "$(CLASSIC_TEST_PATH)" "$(CLASSIC_CONFIG_PATH)" $(ALPHA) 

run_print_prosper:
	$(PYTHON_INTERPRETER) $(SRC_DIR)visualization.print_result  $(DB_NAME)	"$(_DIR)"  $(DISCRETIZATION_TYPE) $(GRAPH_TYPE)

run_plot_prosper:
	$(PYTHON_INTERPRETER) $(SRC_DIR)build_shap_plot.py $(TARGET_NAME_PROSPER)  $(DB_NAME)  "$(TRAIN_DESCRIPTORS_PATHS)" "$(TEST_DESCRIPTORS_PATHS)"  $(MODEL) $(DISCRETIZATION_TYPE)

run_summarize_prosper:
	$(PYTHON_INTERPRETER) $(SRC_DIR)summarize_shap_attributes_importance.py $(DB_NAME) $(DISCRETIZATION_TYPE)

run_without_normalization_prosper:
	$(PYTHON_INTERPRETER) $(SRC_DIR)launchers.without_normalization.launch_per_database $(DB_NAME)

run_with_normalization_prosper:
	$(PYTHON_INTERPRETER) $(SRC_DIR)launchers.with_normalization.launch_per_database  $(DB_NAME)




#SME

########################################################### DATA PREPROCESSING COMMANDS #######################################################################################

run_splitting_sme:
	$(PYTHON_INTERPRETER) $(PRE_DIR)splitting $(DB_NAME) $(DB_PATH_SME)	$(TARGET_NAME_SME) $(USELESS_ATTRIBUTES_SME)  $(TARGET_VALUES_SME)

run_engine_building_pre_sme:
	$(PYTHON_INTERPRETER) $(PRE_DIR)build_preprocess_engine $(DB_NAME) $(TRAINSET_PATH_SME) $(TARGET_NAME_SME)

run_engine_building_unsupervised_discretization_sme:
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.engine.build_unsupervised_discretization_engine $(DB_NAME) $(TARGET_NAME_SME) "$(_PATH)"  "$(_DIR)" 

run_engine_building_supervised_discretization_sme:
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.engine.build_supervised_discretization_engine $(DB_NAME) $(TARGET_NAME_SME) "$(_PATH)" "$(_DIR)" 

run_preprocess_train_sme:
	$(PYTHON_INTERPRETER) $(PRE_DIR)preprocessing $(DB_NAME) $(TRAINSET_PATH_SME) $(TARGET_NAME_SME) $(USELESS_ATTRIBUTES_SME)  $(TRAIN_LABEL)

run_preprocess_test_sme:
	$(PYTHON_INTERPRETER) $(PRE_DIR)preprocessing $(DB_NAME) $(TESTSET_PATH_SME) $(TARGET_NAME_SME) $(USELESS_ATTRIBUTES_SME)  $(TEST_LABEL)

run_preprocess_sme:
	$(PYTHON_INTERPRETER) $(PRE_DIR)preprocessing_of_state_of_art $(DB_NAME) $(DB_PATH_SME) $(TARGET_NAME_SME) $(USELESS_ATTRIBUTES_SME)  

run_supervised_discretization_sme:
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.supervised_discretization_process  $(DB_NAME)  $(TARGET_NAME_SME)  "$(_PATH)"  $(NORMALIZATION_LABEL)  $(DATA_LABEL) 

run_unsupervised_discretization_sme:
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.unsupervised_discretization_process  $(DB_NAME)  $(TARGET_NAME_SME)  "$(_PATH)"  $(NORMALIZATION_LABEL)  $(DATA_LABEL) 

	
########################################################################### GRAPH MANAGEMENT COMMANDS #########################################################################

# BIP GRAPH	
run_graph_modeling_bip_sme:
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.graph.bip.build_graph $(DB_NAME) $(TARGET_NAME_SME)  $(GRAPH_TYPE) $(DISCRETIZATION_TYPE) "$(TRAIN_PATH)" "$(TEST_PATH)"  "$(_DIR)"

run_compute_descriptors_bip_sme: 
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.graph.bip.compute_descriptors $(TARGET_NAME_SME)  $(BD_NAME) $(GRAPH_TYPE) $(ALPHA)  $(DISCRETIZATION_TYPE) "$(TRAIN_PATH)" "$(TEST_PATH)"  "$(_DIR)"  "$(GRAPH_DIR)"


# MOD GRAPH	 
run_graph_modeling_mod_sme:
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.graph.mod.build_graph $(DB_NAME) $(TARGET_NAME_SME)  $(GRAPH_TYPE) $(DISCRETIZATION_TYPE) "$(TRAIN_PATH)" "$(TEST_PATH)"  "$(_DIR)"

run_compute_descriptors_mod_sme: 
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.graph.mod.compute_descriptors $(TARGET_NAME_SME)  $(BD_NAME) $(GRAPH_TYPE) $(ALPHA)  $(DISCRETIZATION_TYPE) "$(TRAIN_PATH)" "$(TEST_PATH)"  "$(_DIR)"  "$(GRAPH_DIR)"



# COMPLETE GRAPHS MANAGEMENT
run_graph_modeling_complete_sme:
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.graph.complete.build_graph $(DB_NAME)  "$(_DIR)"

run_create_edges_of_train_sme: 	
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.graph.complete.create_edges_of_train $(DB_NAME) $(START) $(END) "$(_PATH)" "$(_DIR)"

run_create_edges_of_test_sme: 	
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.graph.complete.create_edges_of_test $(DB_NAME) $(START) $(END) "$(_PATH)" "$(_DIR)"

run_relate_edges_of_train_sme: 	
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.graph.complete.relate_edges_of_train $(DB_NAME) $(START1) $(END1) $(START2) $(END2) "$(_PATH)"  "$(_DIR)"

run_relate_edges_of_test_sme: 	
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.graph.complete.relate_edges_of_test $(DB_NAME) $(START1) $(END1) $(START2) $(END2) "$(_PATH)"  "$(_DIR)"

run_relate_edges_of_both_sme: 	
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.graph.complete.relate_edges_of_both $(DB_NAME) $(START1) $(END1) $(START2) $(END2) "$(TRAIN_PATH)" "$(TEST_PATH)" "$(_DIR)"

run_compute_descriptors_liu_v1_sme: 
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.graph.liu.compute_descriptors_v1 $(TARGET_NAME_SME)  $(BD_NAME) $(GRAPH_TYPE) "$(TRAIN_PATH)"  "$(TEST_PATH)"  "$(_DIR)"  "$(GRAPH_DIR)"

run_compute_descriptors_liu_v2_sme: 
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.graph.liu.compute_descriptors_v2 $(TARGET_NAME_SME)  $(BD_NAME) $(GRAPH_TYPE) "$(TRAIN_PATH)"  "$(TEST_PATH)"  "$(_DIR)"  "$(GRAPH_DIR)"

run_compute_descriptors_gui_sme: 
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.graph.gui.compute_descriptors $(TARGET_NAME_SME)  $(BD_NAME) $(GRAPH_TYPE) "$(TRAIN_PATH)"  "$(TEST_PATH)"  "$(_DIR)"  "$(GRAPH_DIR)" 

run_graph_modeling_gui_sme:
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.graph.gui.build_graph $(DB_NAME) $(TARGET_NAME_SME)  $(GRAPH_TYPE) $(DISCRETIZATION_TYPE) "$(TRAIN_PATH)" "$(TEST_PATH)"  "$(_DIR)"

run_graph_modeling_liu_sme:
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.graph.liu.build_graph $(DB_NAME) $(TARGET_NAME_SME)  $(GRAPH_TYPE) $(DISCRETIZATION_TYPE) "$(TRAIN_PATH)" "$(TEST_PATH)"  "$(_DIR)"	

# GLO GRAPH
run_compute_descriptors_glo_sme:
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.graph.complete.glo.compute_descriptors  $(TARGET_NAME_SME)  $(BD_NAME) $(GRAPH_TYPE) $(ALPHA)	

run_graph_glo_sme:
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.graph.complete.glo.launch  $(DB_NAME)	

# LOAN GRAPH	
run_graph_modeling_loan_sme:
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.graph.loan.build_graph $(DB_NAME) $(TARGET_NAME_SME) $(GRAPH_TYPE) "$(TRAIN_PATH)"  "$(TEST_PATH)"  "$(_DIR)"

run_compute_descriptors_loan_sme: 
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.graph.loan.compute_descriptors $(TARGET_NAME_SME)  $(BD_NAME)  $(GRAPH_TYPE)  $(ALPHA)  "$(TRAIN_PATH)"  "$(TEST_PATH)"  "$(GRAPH_DIR)"  "$(_DIR)"

###############################################################################################################################################################

run_make_configurations_sme:
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.build_configurations $(TARGET_NAME_SME) $(DB_NAME) $(SAVE_DIR) $(DISCRETIZATION_TYPE) $(GRAPH_TYPE) $(CLASSIC_TRAIN_PATH) $(NEW_DESCRIPTOR_TRAIN_PATH) 

run_make_configurations_with_stepwise_sme:
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.build_configurations_with_stepwise $(TARGET_NAME_SME) $(DB_NAME) $(SAVE_DIR) $(DISCRETIZATION_TYPE) $(GRAPH_TYPE) $(CLASSIC_TRAIN_PATH) $(NEW_DESCRIPTOR_TRAIN_PATH) 

run_select_features_sme:
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.select_features_with_pvalue $(TARGET_NAME_SME) $(DB_NAME) "$(TRAIN_PATH)" "$(SAVE_DIR)"
	
run_make_predictions_sme:
	$(PYTHON_INTERPRETER) $(SRC_DIR)models.make_predictions_main $(TARGET_NAME_SME) $(DB_NAME)  "$(TRAIN_PATH)"  "$(TEST_PATH)"  $(DISCRETIZATION_TYPE) $(GRAPH_TYPE) "$(CONFIG_PATH)" "$(SAVE_DIR)"  "$(CLASSIC_TRAIN_PATH)"  "$(CLASSIC_TEST_PATH)" "$(CLASSIC_CONFIG_PATH)" $(ALPHA) 

run_print_sme:
	$(PYTHON_INTERPRETER) $(SRC_DIR)visualization.print_result  $(DB_NAME)	"$(_DIR)"  $(DISCRETIZATION_TYPE) $(GRAPH_TYPE)

run_plot_sme:
	$(PYTHON_INTERPRETER) $(SRC_DIR)build_shap_plot.py $(TARGET_NAME_SME)  $(DB_NAME)  "$(TRAIN_DESCRIPTORS_PATHS)" "$(TEST_DESCRIPTORS_PATHS)"  $(MODEL) $(DISCRETIZATION_TYPE)

run_summarize_sme:
	$(PYTHON_INTERPRETER) $(SRC_DIR)summarize_shap_attributes_importance.py $(DB_NAME) $(DISCRETIZATION_TYPE)

run_without_normalization_sme:
	$(PYTHON_INTERPRETER) $(SRC_DIR)launchers.without_normalization.launch_per_database $(DB_NAME)

run_with_normalization_sme:
	$(PYTHON_INTERPRETER) $(SRC_DIR)launchers.with_normalization.launch_per_database  $(DB_NAME)













#GERMAN
run_preprocess_train_german:
	$(PYTHON_INTERPRETER) $(PRE_DIR)preprocessing $(DB_NAME) $(TRAINSET_PATH_GERMAN) $(TARGET_NAME_GERMAN) $(USELESS_ATTRIBUTES_GERMAN)  $(TRAIN_LABEL)

run_preprocess_test_german:
	$(PYTHON_INTERPRETER) $(PRE_DIR)preprocessing $(DB_NAME) $(TESTSET_PATH_GERMAN) $(TARGET_NAME_GERMAN) $(USELESS_ATTRIBUTES_GERMAN)  $(TEST_LABEL)

run_discretization_german:
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.discretization  $(DB_NAME) $(DISCRETIZATION_TYPE) $(LABEL) $(TARGET_NAME_GERMAN)

run_graph_modeling_german:
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.graph.modeling $(DB_NAME) $(TARGET_NAME_GERMAN)  $(GRAPH_TYPE) $(DISCRETIZATION_TYPE)

run_compute_descriptors_german:
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.graph.compute_descriptors $(TARGET_NAME_GERMAN)  $(BD_NAME) $(ALPHA) $(GRAPH_TYPE) $(DISCRETIZATION_TYPE) $(LABEL)

run_make_configurations_german:
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.build_configurations $(TARGET_NAME_GERMAN) $(DB_NAME)  $(DISCRETIZATION_TYPE) $(GRAPH_TYPE)

run_make_predictions_german:
	$(PYTHON_INTERPRETER) $(SRC_DIR)models.make_predictions_main $(TARGET_NAME_GERMAN) $(DB_NAME)  "$(TRAIN_PATH)"  "$(TEST_PATH)"  $(DISCRETIZATION_TYPE) $(GRAPH_TYPE) $(CONFIG_PATH) $(ALPHA)

run_print_german:
	$(PYTHON_INTERPRETER) $(SRC_DIR)visualization.print_result $(DB_NAME)	$(DISCRETIZATION_TYPE) $(GRAPH_TYPE) 

run_plot_german:
	$(PYTHON_INTERPRETER) $(SRC_DIR)build_shap_plot $(TARGET_NAME_GERMAN)  $(DB_NAME)  "$(TRAIN_DESCRIPTORS_PATHS)" "$(TEST_DESCRIPTORS_PATHS)"  $(MODEL) $(DISCRETIZATION_TYPE)

run_summarize_german:
	$(PYTHON_INTERPRETER) $(SRC_DIR)summarize_shap_attributes_importance.py $(DB_NAME) $(DISCRETIZATION_TYPE)

run_splitting_german:
	$(PYTHON_INTERPRETER) $(PRE_DIR)splitting $(DB_NAME) $(DB_PATH_GERMAN)  $(TARGET_NAME_GERMAN) $(USELESS_ATTRIBUTES_GERMAN)  $(TARGET_VALUES_GERMAN)

run_engine_building_pre_german: 
	$(PYTHON_INTERPRETER) $(PRE_DIR)build_preprocess_engine $(DB_NAME) $(TRAINSET_PATH_GERMAN) $(TARGET_NAME_GERMAN)

run_engine_building_disc_german:
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.build_discretization_engine $(DB_NAME)   $(DISCRETIZATION_TYPE) $(TARGET_NAME_GERMAN)

run_german:
	$(PYTHON_INTERPRETER) $(SRC_DIR)single_launch  $(DB_NAME)



# HMEQ

########################################################### DATA PREPROCESSING COMMANDS #######################################################################################

run_splitting_hmeq:
	$(PYTHON_INTERPRETER) $(PRE_DIR)splitting $(DB_NAME) $(DB_PATH_HMEQ)	$(TARGET_NAME_HMEQ) $(USELESS_ATTRIBUTES_HMEQ)  $(TARGET_VALUES_HMEQ)

run_engine_building_pre_hmeq:
	$(PYTHON_INTERPRETER) $(PRE_DIR)build_preprocess_engine $(DB_NAME) $(TRAINSET_PATH_HMEQ) $(TARGET_NAME_HMEQ)

run_engine_building_unsupervised_discretization_hmeq:
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.engine.build_unsupervised_discretization_engine $(DB_NAME) $(TARGET_NAME_HMEQ) "$(_PATH)"  "$(_DIR)" 

run_engine_building_supervised_discretization_hmeq:
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.engine.build_supervised_discretization_engine $(DB_NAME) $(TARGET_NAME_HMEQ) "$(_PATH)" "$(_DIR)" 

run_preprocess_train_hmeq:
	$(PYTHON_INTERPRETER) $(PRE_DIR)preprocessing $(DB_NAME) $(TRAINSET_PATH_HMEQ) $(TARGET_NAME_HMEQ) $(USELESS_ATTRIBUTES_HMEQ)  $(TRAIN_LABEL)

run_preprocess_test_hmeq:
	$(PYTHON_INTERPRETER) $(PRE_DIR)preprocessing $(DB_NAME) $(TESTSET_PATH_HMEQ) $(TARGET_NAME_HMEQ) $(USELESS_ATTRIBUTES_HMEQ)  $(TEST_LABEL)	

run_supervised_discretization_hmeq:
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.supervised_discretization_process  $(DB_NAME)  $(TARGET_NAME_HMEQ)  "$(_PATH)"  $(NORMALIZATION_LABEL)  $(DATA_LABEL) 

run_unsupervised_discretization_hmeq:
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.unsupervised_discretization_process  $(DB_NAME)  $(TARGET_NAME_HMEQ)  "$(_PATH)"  $(NORMALIZATION_LABEL)  $(DATA_LABEL) 

	
########################################################################### GRAPH MANAGEMENT COMMANDS #########################################################################

# BIP GRAPH	
run_graph_modeling_bip_hmeq:
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.graph.bip.build_graph $(DB_NAME) $(TARGET_NAME_HMEQ)  $(GRAPH_TYPE) $(DISCRETIZATION_TYPE) "$(TRAIN_PATH)" "$(TEST_PATH)"  "$(_DIR)"

run_compute_descriptors_bip_hmeq: 
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.graph.bip.compute_descriptors $(TARGET_NAME_HMEQ)  $(BD_NAME) $(GRAPH_TYPE) $(ALPHA)  $(DISCRETIZATION_TYPE) "$(TRAIN_PATH)" "$(TEST_PATH)"  "$(_DIR)"  "$(GRAPH_DIR)"


# MOD GRAPH	 
run_graph_modeling_mod_hmeq:
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.graph.mod.build_graph $(DB_NAME) $(TARGET_NAME_HMEQ)  $(GRAPH_TYPE) $(DISCRETIZATION_TYPE) "$(TRAIN_PATH)" "$(TEST_PATH)"  "$(_DIR)"

run_compute_descriptors_mod_hmeq: 
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.graph.mod.compute_descriptors $(TARGET_NAME_HMEQ)  $(BD_NAME) $(GRAPH_TYPE) $(ALPHA)  $(DISCRETIZATION_TYPE) "$(TRAIN_PATH)" "$(TEST_PATH)"  "$(_DIR)"  "$(GRAPH_DIR)"



# COMPLETE GRAPHS MANAGEMENT
run_graph_modeling_complete_hmeq:
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.graph.complete.build_graph $(DB_NAME)  "$(_DIR)"

run_create_edges_of_train_hmeq: 	
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.graph.complete.create_edges_of_train $(DB_NAME) $(START) $(END) "$(_PATH)" "$(_DIR)"

run_create_edges_of_test_hmeq: 	
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.graph.complete.create_edges_of_test $(DB_NAME) $(START) $(END) "$(_PATH)" "$(_DIR)"

run_relate_edges_of_train_hmeq: 	
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.graph.complete.relate_edges_of_train $(DB_NAME) $(START1) $(END1) $(START2) $(END2) "$(_PATH)"  "$(_DIR)"

run_relate_edges_of_test_hmeq: 	
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.graph.complete.relate_edges_of_test $(DB_NAME) $(START1) $(END1) $(START2) $(END2) "$(_PATH)"  "$(_DIR)"

run_relate_edges_of_both_hmeq: 	
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.graph.complete.relate_edges_of_both $(DB_NAME) $(START1) $(END1) $(START2) $(END2) "$(TRAIN_PATH)" "$(TEST_PATH)" "$(_DIR)"

run_compute_descriptors_liu_hmeq: 
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.graph.complete.liu.compute_descriptors $(TARGET_NAME_HMEQ)  $(BD_NAME) $(GRAPH_TYPE) "$(TRAIN_PATH)"  "$(TEST_PATH)"  "$(_DIR)"  "$(GRAPH_DIR)"

run_compute_descriptors_gui_hmeq: 
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.graph.complete.gui.compute_descriptors $(TARGET_NAME_HMEQ)  $(BD_NAME) $(GRAPH_TYPE) "$(TRAIN_PATH)"  "$(TEST_PATH)"  "$(_DIR)"  "$(GRAPH_DIR)" 


# GLO GRAPH
run_compute_descriptors_glo_hmeq:
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.graph.complete.glo.compute_descriptors  $(TARGET_NAME_HMEQ)  $(BD_NAME) $(GRAPH_TYPE) $(ALPHA)	

run_graph_glo_hmeq:
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.graph.complete.glo.launch  $(DB_NAME)	

# LOAN GRAPH	
run_graph_modeling_loan_hmeq:
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.graph.loan.build_graph $(DB_NAME) $(TARGET_NAME_HMEQ) $(GRAPH_TYPE) "$(TRAIN_PATH)"  "$(TEST_PATH)"  "$(_DIR)"

run_compute_descriptors_loan_hmeq: 
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.graph.loan.compute_descriptors $(TARGET_NAME_HMEQ)  $(BD_NAME)  $(GRAPH_TYPE)  $(ALPHA)  "$(TRAIN_PATH)"  "$(TEST_PATH)"  "$(GRAPH_DIR)"  "$(_DIR)"

###############################################################################################################################################################

run_make_configurations_hmeq:
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.build_configurations $(TARGET_NAME_HMEQ) $(DB_NAME) $(SAVE_DIR) $(DISCRETIZATION_TYPE) $(GRAPH_TYPE) $(CLASSIC_TRAIN_PATH) $(NEW_DESCRIPTOR_TRAIN_PATH) 

run_make_configurations_with_stepwise_hmeq:
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.build_configurations_with_stepwise $(TARGET_NAME_HMEQ) $(DB_NAME) $(SAVE_DIR) $(DISCRETIZATION_TYPE) $(GRAPH_TYPE) $(CLASSIC_TRAIN_PATH) $(NEW_DESCRIPTOR_TRAIN_PATH) 

run_select_features_hmeq:
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.select_features_with_pvalue $(TARGET_NAME_HMEQ) $(DB_NAME) "$(TRAIN_PATH)" "$(SAVE_DIR)"
	
run_make_predictions_hmeq:
	$(PYTHON_INTERPRETER) $(SRC_DIR)models.make_predictions_main $(TARGET_NAME_HMEQ) $(DB_NAME)  "$(TRAIN_PATH)"  "$(TEST_PATH)"  $(DISCRETIZATION_TYPE) $(GRAPH_TYPE) "$(CONFIG_PATH)" "$(SAVE_DIR)"  "$(CLASSIC_TRAIN_PATH)"  "$(CLASSIC_TEST_PATH)" "$(CLASSIC_CONFIG_PATH)" $(ALPHA) 

run_print_hmeq:
	$(PYTHON_INTERPRETER) $(SRC_DIR)visualization.print_result  $(DB_NAME)	"$(_DIR)"  $(DISCRETIZATION_TYPE) $(GRAPH_TYPE)

run_plot_hmeq:
	$(PYTHON_INTERPRETER) $(SRC_DIR)build_shap_plot.py $(TARGET_NAME_HMEQ)  $(DB_NAME)  "$(TRAIN_DESCRIPTORS_PATHS)" "$(TEST_DESCRIPTORS_PATHS)"  $(MODEL) $(DISCRETIZATION_TYPE)

run_summarize_hmeq:
	$(PYTHON_INTERPRETER) $(SRC_DIR)summarize_shap_attributes_importance.py $(DB_NAME) $(DISCRETIZATION_TYPE)

run_without_normalization_hmeq:
	$(PYTHON_INTERPRETER) $(SRC_DIR)launchers.without_normalization.launch_per_database $(DB_NAME)

run_with_normalization_hmeq:
	$(PYTHON_INTERPRETER) $(SRC_DIR)launchers.with_normalization.launch_per_database  $(DB_NAME)


#KAGGLE-CREDIT-RISK
run_preprocess_train_kaggle_credit_risk:
	$(PYTHON_INTERPRETER) $(PRE_DIR)preprocessing $(DB_NAME) $(TRAINSET_PATH_KAGGLE_CREDIT_RISK) $(TARGET_NAME_KAGGLE_CREDIT_RISK) $(USELESS_ATTRIBUTES_KAGGLE_CREDIT_RISK)  $(TRAIN_LABEL)

run_preprocess_test_kaggle_credit_risk:
	$(PYTHON_INTERPRETER) $(PRE_DIR)preprocessing $(DB_NAME) $(TESTSET_PATH_KAGGLE_CREDIT_RISK) $(TARGET_NAME_KAGGLE_CREDIT_RISK) $(USELESS_ATTRIBUTES_KAGGLE_CREDIT_RISK)  $(TEST_LABEL)
	
run_discretization_kaggle_credit_risk:
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.discretization  $(DB_NAME) $(DISCRETIZATION_TYPE) $(LABEL) $(TARGET_NAME_KAGGLE_CREDIT_RISK)

run_graph_modeling_kaggle_credit_risk:
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.graph.modeling $(DB_NAME) $(TARGET_NAME_KAGGLE_CREDIT_RISK) $(GRAPH_TYPE) $(DISCRETIZATION_TYPE) 
	
run_compute_descriptors_kaggle_credit_risk:
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.graph.compute_descriptors $(TARGET_NAME_KAGGLE_CREDIT_RISK)  $(BD_NAME) $(ALPHA) $(GRAPH_TYPE) $(DISCRETIZATION_TYPE) $(LABEL)

run_make_configurations_kaggle_credit_risk:
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.build_configurations $(TARGET_NAME_KAGGLE_CREDIT_RISK) $(DB_NAME)  $(DISCRETIZATION_TYPE) $(GRAPH_TYPE)

run_make_predictions_kaggle_credit_risk:
	$(PYTHON_INTERPRETER) $(SRC_DIR)models.make_predictions_main $(TARGET_NAME_KAGGLE_CREDIT_RISK) $(DB_NAME)  "$(TRAIN_PATH)"  "$(TEST_PATH)"  $(DISCRETIZATION_TYPE) $(GRAPH_TYPE) $(CONFIG_PATH) $(ALPHA)

run_print_kaggle_credit_risk:
	$(PYTHON_INTERPRETER) $(SRC_DIR)visualization.print_result $(DB_NAME)	$(DISCRETIZATION_TYPE) $(GRAPH_TYPE) 

run_plot_kaggle_credit_risk:
	$(PYTHON_INTERPRETER) $(SRC_DIR)build_shap_plot $(TARGET_NAME_KAGGLE_CREDIT_RISK)  $(DB_NAME)  "$(TRAIN_DESCRIPTORS_PATHS)" "$(TEST_DESCRIPTORS_PATHS)"  $(MODEL) $(DISCRETIZATION_TYPE)
	
run_summarize_kaggle_credit_risk:
	$(PYTHON_INTERPRETER) $(SRC_DIR)summarize_shap_attributes_importance.py $(DB_NAME) $(DISCRETIZATION_TYPE)

run_splitting_kaggle_credit_risk:
	$(PYTHON_INTERPRETER) $(PRE_DIR)splitting $(DB_NAME) $(DB_PATH_KAGGLE_CREDIT_RISK)  $(TARGET_NAME_KAGGLE_CREDIT_RISK) $(USELESS_ATTRIBUTES_KAGGLE_CREDIT_RISK)  $(TARGET_VALUES_KAGGLE_CREDIT_RISK)

run_engine_building_pre_kaggle_credit_risk: 
	$(PYTHON_INTERPRETER) $(PRE_DIR)build_preprocess_engine $(DB_NAME) $(TRAINSET_PATH_KAGGLE_CREDIT_RISK) $(TARGET_NAME_KAGGLE_CREDIT_RISK)

run_engine_building_disc_kaggle_credit_risk:
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.build_discretization_engine $(DB_NAME)   $(DISCRETIZATION_TYPE) $(TARGET_NAME_KAGGLE_CREDIT_RISK)

run_kaggle_credit_risk:
	$(PYTHON_INTERPRETER) $(SRC_DIR)single_launch  $(DB_NAME)

run_build_edges_kaggle_credit_risk: 	
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.graph.build_complete_graph $(DB_NAME) $(START) $(END) $(TYPE)

run_relate_edges_kaggle_credit_risk: 	
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.graph.relate_edges_complete_graph $(DB_NAME) $(START1) $(END1)	$(START2) $(END2) $(TYPE)  "$(PATH1)"  "$(PATH2)"


#LGD
run_preprocess_train_lgd:
	$(PYTHON_INTERPRETER) $(PRE_DIR)preprocessing $(DB_NAME) $(TRAINSET_PATH_LGD) $(TARGET_NAME_LGD) $(USELESS_ATTRIBUTES_LGD)  $(TRAIN_LABEL)

run_preprocess_test_lgd:
	$(PYTHON_INTERPRETER) $(PRE_DIR)preprocessing $(DB_NAME) $(TESTSET_PATH_LGD) $(TARGET_NAME_LGD) $(USELESS_ATTRIBUTES_LGD)  $(TEST_LABEL)

run_discretization_lgd:
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.discretization  $(DB_NAME) $(DISCRETIZATION_TYPE) $(LABEL) $(TARGET_NAME_LGD)

run_graph_modeling_lgd:
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.graph.modeling $(DB_NAME) $(TARGET_NAME_LGD)  $(GRAPH_TYPE) $(DISCRETIZATION_TYPE)

run_compute_descriptors_lgd:
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.graph.compute_descriptors $(TARGET_NAME_LGD)  $(BD_NAME) $(ALPHA) $(GRAPH_TYPE) $(DISCRETIZATION_TYPE) $(LABEL)

run_make_configurations_lgd:
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.build_configurations $(TARGET_NAME_LGD) $(DB_NAME)  $(DISCRETIZATION_TYPE) $(GRAPH_TYPE)

run_make_predictions_lgd:
	$(PYTHON_INTERPRETER) $(SRC_DIR)models.make_predictions_main $(TARGET_NAME_LGD) $(DB_NAME)  "$(TRAIN_PATH)"  "$(TEST_PATH)"  $(DISCRETIZATION_TYPE) $(GRAPH_TYPE) $(CONFIG_PATH) $(ALPHA)

run_print_lgd:
	$(PYTHON_INTERPRETER) $(SRC_DIR)visualization.print_result $(DB_NAME)	$(DISCRETIZATION_TYPE) $(GRAPH_TYPE) 

run_plot_lgd:
	$(PYTHON_INTERPRETER) $(SRC_DIR)build_shap_plot $(TARGET_NAME_LGD)  $(DB_NAME)  "$(TRAIN_DESCRIPTORS_PATHS)" "$(TEST_DESCRIPTORS_PATHS)"  $(MODEL) $(DISCRETIZATION_TYPE)

run_summarize_lgd:
	$(PYTHON_INTERPRETER) $(SRC_DIR)summarize_shap_attributes_importance.py $(DB_NAME) $(DISCRETIZATION_TYPE)

run_splitting_lgd:
	$(PYTHON_INTERPRETER) $(PRE_DIR)splitting $(DB_NAME) $(DB_PATH_LGD)  $(TARGET_NAME_LGD) $(USELESS_ATTRIBUTES_LGD)  $(TARGET_VALUES_LGD)

run_engine_building_pre_lgd: 
	$(PYTHON_INTERPRETER) $(PRE_DIR)build_preprocess_engine $(DB_NAME) $(TRAINSET_PATH_LGD) $(TARGET_NAME_LGD)

run_engine_building_disc_lgd:
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.build_discretization_engine $(DB_NAME)   $(DISCRETIZATION_TYPE) $(TARGET_NAME_LGD)

run_lgd:
	$(PYTHON_INTERPRETER) $(SRC_DIR)single_launch  $(DB_NAME)



#AER
run_preprocess_train_aer:
	$(PYTHON_INTERPRETER) $(PRE_DIR)preprocessing $(DB_NAME) $(TRAINSET_PATH_AER) $(TARGET_NAME_AER) $(USELESS_ATTRIBUTES_AER)  $(TRAIN_LABEL)

run_preprocess_test_aer:
	$(PYTHON_INTERPRETER) $(PRE_DIR)preprocessing $(DB_NAME) $(TESTSET_PATH_AER) $(TARGET_NAME_AER) $(USELESS_ATTRIBUTES_AER)  $(TEST_LABEL)

run_discretization_aer:
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.discretization  $(DB_NAME) $(DISCRETIZATION_TYPE) $(LABEL) $(TARGET_NAME_AER)

run_graph_modeling_aer:
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.graph.modeling $(DB_NAME) $(TARGET_NAME_AER)  $(GRAPH_TYPE) $(DISCRETIZATION_TYPE)

run_compute_descriptors_aer:
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.graph.compute_descriptors $(TARGET_NAME_AER)  $(BD_NAME) $(ALPHA) $(GRAPH_TYPE) $(DISCRETIZATION_TYPE) $(LABEL)

run_make_configurations_aer:
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.build_configurations $(TARGET_NAME_AER) $(DB_NAME)  $(DISCRETIZATION_TYPE) $(GRAPH_TYPE)

run_make_predictions_aer:
	$(PYTHON_INTERPRETER) $(SRC_DIR)models.make_predictions_main $(TARGET_NAME_AER) $(DB_NAME)  "$(TRAIN_PATH)"  "$(TEST_PATH)"  $(DISCRETIZATION_TYPE) $(GRAPH_TYPE) $(CONFIG_PATH) $(ALPHA)

run_print_aer:
	$(PYTHON_INTERPRETER) $(SRC_DIR)visualization.print_result $(DB_NAME)	$(DISCRETIZATION_TYPE) $(GRAPH_TYPE) 

run_plot_aer:
	$(PYTHON_INTERPRETER) $(SRC_DIR)build_shap_plot $(TARGET_NAME_AER)  $(DB_NAME)  "$(TRAIN_DESCRIPTORS_PATHS)" "$(TEST_DESCRIPTORS_PATHS)"  $(MODEL) $(DISCRETIZATION_TYPE)

run_summarize_aer:
	$(PYTHON_INTERPRETER) $(SRC_DIR)summarize_shap_attributes_importance.py $(DB_NAME) $(DISCRETIZATION_TYPE)

run_splitting_aer:
	$(PYTHON_INTERPRETER) $(PRE_DIR)splitting $(DB_NAME) $(DB_PATH_AER)  $(TARGET_NAME_AER) $(USELESS_ATTRIBUTES_AER)  $(TARGET_VALUES_AER)

run_engine_building_pre_aer: 
	$(PYTHON_INTERPRETER) $(PRE_DIR)build_preprocess_engine $(DB_NAME) $(TRAINSET_PATH_AER) $(TARGET_NAME_AER)

run_engine_building_disc_aer:
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.build_discretization_engine $(DB_NAME)   $(DISCRETIZATION_TYPE) $(TARGET_NAME_AER)

run_aer:
	$(PYTHON_INTERPRETER) $(SRC_DIR)single_launch  $(DB_NAME)



#THOMAS
run_preprocess_train_thomas:
	$(PYTHON_INTERPRETER) $(PRE_DIR)preprocessing $(DB_NAME) $(TRAINSET_PATH_THOMAS) $(TARGET_NAME_THOMAS) $(USELESS_ATTRIBUTES_THOMAS)  $(TRAIN_LABEL)

run_preprocess_test_thomas:
	$(PYTHON_INTERPRETER) $(PRE_DIR)preprocessing $(DB_NAME) $(TESTSET_PATH_THOMAS) $(TARGET_NAME_THOMAS) $(USELESS_ATTRIBUTES_THOMAS)  $(TEST_LABEL)

run_discretization_thomas:
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.discretization  $(DB_NAME) $(DISCRETIZATION_TYPE) $(LABEL) $(TARGET_NAME_THOMAS)

run_graph_modeling_thomas:
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.graph.modeling $(DB_NAME) $(TARGET_NAME_THOMAS)  $(GRAPH_TYPE) $(DISCRETIZATION_TYPE)

run_compute_descriptors_thomas:
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.graph.compute_descriptors $(TARGET_NAME_THOMAS)  $(BD_NAME) $(ALPHA) $(GRAPH_TYPE) $(DISCRETIZATION_TYPE) $(LABEL)

run_make_configurations_thomas:
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.build_configurations $(TARGET_NAME_THOMAS) $(DB_NAME)  $(DISCRETIZATION_TYPE) $(GRAPH_TYPE)

run_make_predictions_thomas:
	$(PYTHON_INTERPRETER) $(SRC_DIR)models.make_predictions_main $(TARGET_NAME_THOMAS) $(DB_NAME)  "$(TRAIN_PATH)"  "$(TEST_PATH)"  $(DISCRETIZATION_TYPE) $(GRAPH_TYPE) $(CONFIG_PATH) $(ALPHA)

run_print_thomas:
	$(PYTHON_INTERPRETER) $(SRC_DIR)visualization.print_result $(DB_NAME)	$(DISCRETIZATION_TYPE) $(GRAPH_TYPE) 

run_plot_thomas:
	$(PYTHON_INTERPRETER) $(SRC_DIR)build_shap_plot $(TARGET_NAME_THOMAS)  $(DB_NAME)  "$(TRAIN_DESCRIPTORS_PATHS)" "$(TEST_DESCRIPTORS_PATHS)"  $(MODEL) $(DISCRETIZATION_TYPE)

run_summarize_thomas:
	$(PYTHON_INTERPRETER) $(SRC_DIR)summarize_shap_attributes_importance.py $(DB_NAME) $(DISCRETIZATION_TYPE)

run_splitting_thomas:
	$(PYTHON_INTERPRETER) $(PRE_DIR)splitting $(DB_NAME) $(DB_PATH_THOMAS)  $(TARGET_NAME_THOMAS) $(USELESS_ATTRIBUTES_THOMAS)  $(TARGET_VALUES_THOMAS)

run_engine_building_pre_thomas: 
	$(PYTHON_INTERPRETER) $(PRE_DIR)build_preprocess_engine $(DB_NAME) $(TRAINSET_PATH_THOMAS) $(TARGET_NAME_THOMAS)

run_engine_building_disc_thomas:
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.build_discretization_engine $(DB_NAME)   $(DISCRETIZATION_TYPE) $(TARGET_NAME_THOMAS)

run_thomas:
	$(PYTHON_INTERPRETER) $(SRC_DIR)single_launch  $(DB_NAME)

#MORTGAGE
run_sampling_mortgage:
	$(PYTHON_INTERPRETER) $(SRC_DIR)sampling_main.py $(DB_NAME) $(DB_PATH_MORTGAGE) $(TARGET_NAME_MORTGAGE) $(USELESS_ATTRIBUTES_MORTGAGE)   $(TARGET_VALUES_MORTGAGE)

run_preprocess_mortgage:
	$(PYTHON_INTERPRETER) $(SRC_DIR)preprocessing.py $(DB_NAME) $(DB_PATH_MORTGAGE) $(TARGET_NAME_MORTGAGE) $(USELESS_ATTRIBUTES_MORTGAGE)  $(TARGET_VALUES_MORTGAGE)

run_preprocess_train_mortgage:
	$(PYTHON_INTERPRETER) $(PRE_DIR)preprocessing $(DB_NAME) $(TRAINSET_PATH_MORTGAGE) $(TARGET_NAME_MORTGAGE) $(USELESS_ATTRIBUTES_MORTGAGE)  $(TRAIN_LABEL)

run_preprocess_test_mortgage:
	$(PYTHON_INTERPRETER) $(PRE_DIR)preprocessing $(DB_NAME) $(TESTSET_PATH_MORTGAGE) $(TARGET_NAME_MORTGAGE) $(USELESS_ATTRIBUTES_MORTGAGE)  $(TEST_LABEL)

run_discretization_mortgage:
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.discretization  $(DB_NAME) $(DISCRETIZATION_TYPE) $(LABEL) $(TARGET_NAME_MORTGAGE)

run_graph_modeling_mortgage:
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.graph.modeling $(DB_NAME) $(TARGET_NAME_MORTGAGE)  $(GRAPH_TYPE) $(DISCRETIZATION_TYPE)

run_compute_descriptors_mortgage:
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.graph.compute_descriptors $(TARGET_NAME_MORTGAGE)  $(BD_NAME) $(ALPHA) $(GRAPH_TYPE) $(DISCRETIZATION_TYPE) $(LABEL)

run_make_configurations_mortgage:
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.build_configurations $(TARGET_NAME_MORTGAGE) $(DB_NAME)  $(DISCRETIZATION_TYPE) $(GRAPH_TYPE)

run_make_predictions_mortgage:
	$(PYTHON_INTERPRETER) $(SRC_DIR)models.make_predictions_main $(TARGET_NAME_MORTGAGE) $(DB_NAME)  "$(TRAIN_PATH)"  "$(TEST_PATH)"  $(DISCRETIZATION_TYPE) $(GRAPH_TYPE) $(CONFIG_PATH) $(ALPHA)

run_print_mortgage:
	$(PYTHON_INTERPRETER) $(SRC_DIR)visualization.print_result $(DB_NAME)	$(DISCRETIZATION_TYPE) $(GRAPH_TYPE) 

run_plot_mortgage:
	$(PYTHON_INTERPRETER) $(SRC_DIR)build_shap_plot $(TARGET_NAME_MORTGAGE)  $(DB_NAME)  "$(TRAIN_DESCRIPTORS_PATHS)" "$(TEST_DESCRIPTORS_PATHS)"  $(MODEL) $(DISCRETIZATION_TYPE)

run_summarize_mortgage:
	$(PYTHON_INTERPRETER) $(SRC_DIR)summarize_shap_attributes_importance.py $(DB_NAME) $(DISCRETIZATION_TYPE)

run_splitting_mortgage:
	$(PYTHON_INTERPRETER) $(PRE_DIR)splitting $(DB_NAME) $(DB_PATH_MORTGAGE)  $(TARGET_NAME_MORTGAGE) $(USELESS_ATTRIBUTES_MORTGAGE)  $(TARGET_VALUES_MORTGAGE)

run_engine_building_pre_mortgage: 
	$(PYTHON_INTERPRETER) $(PRE_DIR)build_preprocess_engine $(DB_NAME) $(TRAINSET_PATH_MORTGAGE) $(TARGET_NAME_MORTGAGE)

run_engine_building_disc_mortgage:
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.build_discretization_engine $(DB_NAME)   $(DISCRETIZATION_TYPE) $(TARGET_NAME_MORTGAGE)

run_mortgage:
	$(PYTHON_INTERPRETER) $(SRC_DIR)single_launch  $(DB_NAME)





  
#JAPANESE

########################################################### DATA PREPROCESSING COMMANDS #######################################################################################

run_splitting_japanese:
	$(PYTHON_INTERPRETER) $(PRE_DIR)splitting $(DB_NAME) $(DB_PATH_JAPANESE)	$(TARGET_NAME_JAPANESE) $(USELESS_ATTRIBUTES_JAPANESE)  $(TARGET_VALUES_JAPANESE)

run_engine_building_pre_japanese:
	$(PYTHON_INTERPRETER) $(PRE_DIR)build_preprocess_engine $(DB_NAME) $(TRAINSET_PATH_JAPANESE) $(TARGET_NAME_JAPANESE)

run_engine_building_unsupervised_discretization_japanese:
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.engine.build_unsupervised_discretization_engine $(DB_NAME) $(TARGET_NAME_JAPANESE) "$(_PATH)"  "$(_DIR)" 

run_engine_building_supervised_discretization_japanese:
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.engine.build_supervised_discretization_engine $(DB_NAME) $(TARGET_NAME_JAPANESE) "$(_PATH)" "$(_DIR)" 

run_preprocess_train_japanese:
	$(PYTHON_INTERPRETER) $(PRE_DIR)preprocessing $(DB_NAME) $(TRAINSET_PATH_JAPANESE) $(TARGET_NAME_JAPANESE) $(USELESS_ATTRIBUTES_JAPANESE)  $(TRAIN_LABEL)

run_preprocess_test_japanese:
	$(PYTHON_INTERPRETER) $(PRE_DIR)preprocessing $(DB_NAME) $(TESTSET_PATH_JAPANESE) $(TARGET_NAME_JAPANESE) $(USELESS_ATTRIBUTES_JAPANESE)  $(TEST_LABEL)	

run_supervised_discretization_japanese:
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.supervised_discretization_process  $(DB_NAME)  $(TARGET_NAME_JAPANESE)  "$(_PATH)"  $(NORMALIZATION_LABEL)  $(DATA_LABEL) 

run_unsupervised_discretization_japanese:
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.unsupervised_discretization_process  $(DB_NAME)  $(TARGET_NAME_JAPANESE)  "$(_PATH)"  $(NORMALIZATION_LABEL)  $(DATA_LABEL) 

	
########################################################################### GRAPH MANAGEMENT COMMANDS #########################################################################

# BIP GRAPH	
run_graph_modeling_bip_japanese:
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.graph.bip.build_graph $(DB_NAME) $(TARGET_NAME_JAPANESE)  $(GRAPH_TYPE) $(DISCRETIZATION_TYPE) "$(TRAIN_PATH)" "$(TEST_PATH)"  "$(_DIR)"

run_compute_descriptors_bip_japanese: 
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.graph.bip.compute_descriptors $(TARGET_NAME_JAPANESE)  $(BD_NAME) $(GRAPH_TYPE) $(ALPHA)  $(DISCRETIZATION_TYPE) "$(TRAIN_PATH)" "$(TEST_PATH)"  "$(_DIR)"  "$(GRAPH_DIR)"


# MOD GRAPH	 
run_graph_modeling_mod_japanese:
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.graph.mod.build_graph $(DB_NAME) $(TARGET_NAME_JAPANESE)  $(GRAPH_TYPE) $(DISCRETIZATION_TYPE) "$(TRAIN_PATH)" "$(TEST_PATH)"  "$(_DIR)"

run_compute_descriptors_mod_japanese: 
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.graph.mod.compute_descriptors $(TARGET_NAME_JAPANESE)  $(BD_NAME) $(GRAPH_TYPE) $(ALPHA)  $(DISCRETIZATION_TYPE) "$(TRAIN_PATH)" "$(TEST_PATH)"  "$(_DIR)"  "$(GRAPH_DIR)"



# COMPLETE GRAPHS MANAGEMENT
run_graph_modeling_complete_japanese:
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.graph.complete.build_graph $(DB_NAME)  "$(_DIR)"

run_create_edges_of_train_japanese: 	
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.graph.complete.create_edges_of_train $(DB_NAME) $(START) $(END) "$(_PATH)" "$(_DIR)"

run_create_edges_of_test_japanese: 	
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.graph.complete.create_edges_of_test $(DB_NAME) $(START) $(END) "$(_PATH)" "$(_DIR)"

run_relate_edges_of_train_japanese: 	
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.graph.complete.relate_edges_of_train $(DB_NAME) $(START1) $(END1) $(START2) $(END2) "$(_PATH)"  "$(_DIR)"

run_relate_edges_of_test_japanese: 	
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.graph.complete.relate_edges_of_test $(DB_NAME) $(START1) $(END1) $(START2) $(END2) "$(_PATH)"  "$(_DIR)"

run_relate_edges_of_both_japanese: 	
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.graph.complete.relate_edges_of_both $(DB_NAME) $(START1) $(END1) $(START2) $(END2) "$(TRAIN_PATH)" "$(TEST_PATH)" "$(_DIR)"

run_compute_descriptors_liu_japanese: 
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.graph.complete.liu.compute_descriptors $(TARGET_NAME_JAPANESE)  $(BD_NAME) $(GRAPH_TYPE) "$(TRAIN_PATH)"  "$(TEST_PATH)"  "$(_DIR)"  "$(GRAPH_DIR)"

run_compute_descriptors_gui_japanese: 
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.graph.complete.gui.compute_descriptors $(TARGET_NAME_JAPANESE)  $(BD_NAME) $(GRAPH_TYPE) "$(TRAIN_PATH)"  "$(TEST_PATH)"  "$(_DIR)"  "$(GRAPH_DIR)" 


# GLO GRAPH
run_compute_descriptors_glo_japanese:
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.graph.complete.glo.compute_descriptors  $(TARGET_NAME_JAPANESE)  $(BD_NAME) $(GRAPH_TYPE) $(ALPHA)	

run_graph_glo_japanese:
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.graph.complete.glo.launch  $(DB_NAME)	

# LOAN GRAPH	
run_graph_modeling_loan_japanese:
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.graph.loan.build_graph $(DB_NAME) $(TARGET_NAME_JAPANESE) $(GRAPH_TYPE) "$(TRAIN_PATH)"  "$(TEST_PATH)"  "$(_DIR)"

run_compute_descriptors_loan_japanese: 
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.graph.loan.compute_descriptors $(TARGET_NAME_JAPANESE)  $(BD_NAME)  $(GRAPH_TYPE)  $(ALPHA)  "$(TRAIN_PATH)"  "$(TEST_PATH)"  "$(GRAPH_DIR)"  "$(_DIR)"

###############################################################################################################################################################

run_make_configurations_japanese:
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.build_configurations $(TARGET_NAME_JAPANESE) $(DB_NAME) $(SAVE_DIR) $(DISCRETIZATION_TYPE) $(GRAPH_TYPE) $(CLASSIC_TRAIN_PATH) $(NEW_DESCRIPTOR_TRAIN_PATH) 

run_make_configurations_with_stepwise_japanese:
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.build_configurations_with_stepwise $(TARGET_NAME_JAPANESE) $(DB_NAME) $(SAVE_DIR) $(DISCRETIZATION_TYPE) $(GRAPH_TYPE) $(CLASSIC_TRAIN_PATH) $(NEW_DESCRIPTOR_TRAIN_PATH) 

run_select_features_japanese:
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.select_features_with_pvalue $(TARGET_NAME_JAPANESE) $(DB_NAME) "$(TRAIN_PATH)" "$(SAVE_DIR)"
	
run_make_predictions_japanese:
	$(PYTHON_INTERPRETER) $(SRC_DIR)models.make_predictions_main $(TARGET_NAME_JAPANESE) $(DB_NAME)  "$(TRAIN_PATH)"  "$(TEST_PATH)"  $(DISCRETIZATION_TYPE) $(GRAPH_TYPE) "$(CONFIG_PATH)" "$(SAVE_DIR)"  "$(CLASSIC_TRAIN_PATH)"  "$(CLASSIC_TEST_PATH)" "$(CLASSIC_CONFIG_PATH)" $(ALPHA) 

run_print_japanese:
	$(PYTHON_INTERPRETER) $(SRC_DIR)visualization.print_result  $(DB_NAME)	"$(_DIR)"  $(DISCRETIZATION_TYPE) $(GRAPH_TYPE)

run_plot_japanese:
	$(PYTHON_INTERPRETER) $(SRC_DIR)build_shap_plot.py $(TARGET_NAME_JAPANESE)  $(DB_NAME)  "$(TRAIN_DESCRIPTORS_PATHS)" "$(TEST_DESCRIPTORS_PATHS)"  $(MODEL) $(DISCRETIZATION_TYPE)

run_summarize_japanese:
	$(PYTHON_INTERPRETER) $(SRC_DIR)summarize_shap_attributes_importance.py $(DB_NAME) $(DISCRETIZATION_TYPE)

run_without_normalization_japanese:
	$(PYTHON_INTERPRETER) $(SRC_DIR)launchers.without_normalization.launch_per_database $(DB_NAME)

run_with_normalization_japanese:
	$(PYTHON_INTERPRETER) $(SRC_DIR)launchers.with_normalization.launch_per_database  $(DB_NAME)





#AUSTRALIAN

########################################################### DATA PREPROCESSING COMMANDS #######################################################################################

run_splitting_australian:
	$(PYTHON_INTERPRETER) $(PRE_DIR)splitting $(DB_NAME) $(DB_PATH_AUSTRALIAN)	$(TARGET_NAME_AUSTRALIAN) $(USELESS_ATTRIBUTES_AUSTRALIAN)  $(TARGET_VALUES_AUSTRALIAN)

run_engine_building_pre_australian:
	$(PYTHON_INTERPRETER) $(PRE_DIR)build_preprocess_engine $(DB_NAME) $(TRAINSET_PATH_AUSTRALIAN) $(TARGET_NAME_AUSTRALIAN)

run_engine_building_unsupervised_discretization_australian:
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.engine.build_unsupervised_discretization_engine $(DB_NAME) $(TARGET_NAME_AUSTRALIAN) "$(_PATH)"  "$(_DIR)" 

run_engine_building_supervised_discretization_australian:
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.engine.build_supervised_discretization_engine $(DB_NAME) $(TARGET_NAME_AUSTRALIAN) "$(_PATH)" "$(_DIR)" 

run_preprocess_train_australian:
	$(PYTHON_INTERPRETER) $(PRE_DIR)preprocessing $(DB_NAME) $(TRAINSET_PATH_AUSTRALIAN) $(TARGET_NAME_AUSTRALIAN) $(USELESS_ATTRIBUTES_AUSTRALIAN)  $(TRAIN_LABEL)

run_preprocess_test_australian:
	$(PYTHON_INTERPRETER) $(PRE_DIR)preprocessing $(DB_NAME) $(TESTSET_PATH_AUSTRALIAN) $(TARGET_NAME_AUSTRALIAN) $(USELESS_ATTRIBUTES_AUSTRALIAN)  $(TEST_LABEL)	

run_supervised_discretization_australian:
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.supervised_discretization_process  $(DB_NAME)  $(TARGET_NAME_AUSTRALIAN)  "$(_PATH)"  $(NORMALIZATION_LABEL)  $(DATA_LABEL) 

run_unsupervised_discretization_australian:
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.unsupervised_discretization_process  $(DB_NAME)  $(TARGET_NAME_AUSTRALIAN)  "$(_PATH)"  $(NORMALIZATION_LABEL)  $(DATA_LABEL) 

	
########################################################################### GRAPH MANAGEMENT COMMANDS #########################################################################

# BIP GRAPH	
run_graph_modeling_bip_australian:
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.graph.bip.build_graph $(DB_NAME) $(TARGET_NAME_AUSTRALIAN)  $(GRAPH_TYPE) $(DISCRETIZATION_TYPE) "$(TRAIN_PATH)" "$(TEST_PATH)"  "$(_DIR)"

run_compute_descriptors_bip_australian: 
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.graph.bip.compute_descriptors $(TARGET_NAME_AUSTRALIAN)  $(BD_NAME) $(GRAPH_TYPE) $(ALPHA)  $(DISCRETIZATION_TYPE) "$(TRAIN_PATH)" "$(TEST_PATH)"  "$(_DIR)"  "$(GRAPH_DIR)"


# MOD GRAPH	 
run_graph_modeling_mod_australian:
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.graph.mod.build_graph $(DB_NAME) $(TARGET_NAME_AUSTRALIAN)  $(GRAPH_TYPE) $(DISCRETIZATION_TYPE) "$(TRAIN_PATH)" "$(TEST_PATH)"  "$(_DIR)"

run_compute_descriptors_mod_australian: 
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.graph.mod.compute_descriptors $(TARGET_NAME_AUSTRALIAN)  $(BD_NAME) $(GRAPH_TYPE) $(ALPHA)  $(DISCRETIZATION_TYPE) "$(TRAIN_PATH)" "$(TEST_PATH)"  "$(_DIR)"  "$(GRAPH_DIR)"



# COMPLETE GRAPHS MANAGEMENT
run_graph_modeling_complete_australian:
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.graph.complete.build_graph $(DB_NAME)  "$(_DIR)"

run_create_edges_of_train_australian: 	
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.graph.complete.create_edges_of_train $(DB_NAME) $(START) $(END) "$(_PATH)" "$(_DIR)"

run_create_edges_of_test_australian: 	
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.graph.complete.create_edges_of_test $(DB_NAME) $(START) $(END) "$(_PATH)" "$(_DIR)"

run_relate_edges_of_train_australian: 	
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.graph.complete.relate_edges_of_train $(DB_NAME) $(START1) $(END1) $(START2) $(END2) "$(_PATH)"  "$(_DIR)"

run_relate_edges_of_test_australian: 	
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.graph.complete.relate_edges_of_test $(DB_NAME) $(START1) $(END1) $(START2) $(END2) "$(_PATH)"  "$(_DIR)"

run_relate_edges_of_both_australian: 	
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.graph.complete.relate_edges_of_both $(DB_NAME) $(START1) $(END1) $(START2) $(END2) "$(TRAIN_PATH)" "$(TEST_PATH)" "$(_DIR)"

run_compute_descriptors_liu_australian: 
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.graph.complete.liu.compute_descriptors $(TARGET_NAME_AUSTRALIAN)  $(BD_NAME) $(GRAPH_TYPE) "$(TRAIN_PATH)"  "$(TEST_PATH)"  "$(_DIR)"  "$(GRAPH_DIR)"

run_compute_descriptors_gui_australian: 
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.graph.complete.gui.compute_descriptors $(TARGET_NAME_AUSTRALIAN)  $(BD_NAME) $(GRAPH_TYPE) "$(TRAIN_PATH)"  "$(TEST_PATH)"  "$(_DIR)"  "$(GRAPH_DIR)" 


# GLO GRAPH
run_compute_descriptors_glo_australian:
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.graph.complete.glo.compute_descriptors  $(TARGET_NAME_AUSTRALIAN)  $(BD_NAME) $(GRAPH_TYPE) $(ALPHA)	

run_graph_glo_australian:
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.graph.complete.glo.launch  $(DB_NAME)	

# LOAN GRAPH	
run_graph_modeling_loan_australian:
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.graph.loan.build_graph $(DB_NAME) $(TARGET_NAME_AUSTRALIAN) $(GRAPH_TYPE) "$(TRAIN_PATH)"  "$(TEST_PATH)"  "$(_DIR)"

run_compute_descriptors_loan_australian: 
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.graph.loan.compute_descriptors $(TARGET_NAME_AUSTRALIAN)  $(BD_NAME)  $(GRAPH_TYPE)  $(ALPHA)  "$(TRAIN_PATH)"  "$(TEST_PATH)"  "$(GRAPH_DIR)"  "$(_DIR)"

###############################################################################################################################################################

run_make_configurations_australian:
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.build_configurations $(TARGET_NAME_AUSTRALIAN) $(DB_NAME) $(SAVE_DIR) $(DISCRETIZATION_TYPE) $(GRAPH_TYPE) $(CLASSIC_TRAIN_PATH) $(NEW_DESCRIPTOR_TRAIN_PATH) 

run_make_configurations_with_stepwise_australian:
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.build_configurations_with_stepwise $(TARGET_NAME_AUSTRALIAN) $(DB_NAME) $(SAVE_DIR) $(DISCRETIZATION_TYPE) $(GRAPH_TYPE) $(CLASSIC_TRAIN_PATH) $(NEW_DESCRIPTOR_TRAIN_PATH) 

run_select_features_australian:
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.select_features_with_pvalue $(TARGET_NAME_AUSTRALIAN) $(DB_NAME) "$(TRAIN_PATH)" "$(SAVE_DIR)"
	
run_make_predictions_australian:
	$(PYTHON_INTERPRETER) $(SRC_DIR)models.make_predictions_main $(TARGET_NAME_AUSTRALIAN) $(DB_NAME)  "$(TRAIN_PATH)"  "$(TEST_PATH)"  $(DISCRETIZATION_TYPE) $(GRAPH_TYPE) "$(CONFIG_PATH)" "$(SAVE_DIR)"  "$(CLASSIC_TRAIN_PATH)"  "$(CLASSIC_TEST_PATH)" "$(CLASSIC_CONFIG_PATH)" $(ALPHA) 

run_print_australian:
	$(PYTHON_INTERPRETER) $(SRC_DIR)visualization.print_result  $(DB_NAME)	"$(_DIR)"  $(DISCRETIZATION_TYPE) $(GRAPH_TYPE)

run_plot_australian:
	$(PYTHON_INTERPRETER) $(SRC_DIR)build_shap_plot.py $(TARGET_NAME_AUSTRALIAN)  $(DB_NAME)  "$(TRAIN_DESCRIPTORS_PATHS)" "$(TEST_DESCRIPTORS_PATHS)"  $(MODEL) $(DISCRETIZATION_TYPE)

run_summarize_australian:
	$(PYTHON_INTERPRETER) $(SRC_DIR)summarize_shap_attributes_importance.py $(DB_NAME) $(DISCRETIZATION_TYPE)

run_without_normalization_australian:
	$(PYTHON_INTERPRETER) $(SRC_DIR)launchers.without_normalization.launch_per_database $(DB_NAME)

run_with_normalization_australian:
	$(PYTHON_INTERPRETER) $(SRC_DIR)launchers.with_normalization.launch_per_database  $(DB_NAME)



































