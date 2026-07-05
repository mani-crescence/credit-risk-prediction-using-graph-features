include .env
export QT_QPA_PLATFORM=offscreen

# Variables
PYTHON_INTERPRETER=  nice -5  python3
SRC_DIR = -m src.
PRE_DIR = -m src.data_processing.

# all: run

run_all_dbs_with_normalization:
	$(PYTHON_INTERPRETER) $(SRC_DIR)launchers.with_normalization.launch_all_databases $(DB_NAME) $(SUB)

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

run: run_all_dbs_with_normalization 	run_summarization




#BONDORA

########################################################### DATA PREPROCESSING COMMANDS #######################################################################################

run_splitting_bondora:
	$(PYTHON_INTERPRETER) $(PRE_DIR)splitting $(DB_NAME) $(DB_PATH_BONDORA)	$(TARGET_NAME_BONDORA) $(USELESS_ATTRIBUTES_BONDORA)  $(TARGET_VALUES_BONDORA)

run_engine_building_pre_bondora:
	$(PYTHON_INTERPRETER) $(PRE_DIR)build_preprocess_engine $(DB_NAME) $(TRAINSET_PATH_BONDORA) $(TARGET_NAME_BONDORA) 

run_engine_building_unsupervised_discretization_bondora:
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.engine.build_unsupervised_discretization_engine $(DB_NAME) $(TARGET_NAME_BONDORA) "$(_PATH)"  "$(_DIR)" 

run_engine_building_supervised_discretization_bondora:
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.engine.build_supervised_discretization_engine $(DB_NAME) $(TARGET_NAME_BONDORA) "$(_PATH)" "$(_DIR)" $(SUB)

run_preprocess_train_bondora:
	$(PYTHON_INTERPRETER) $(PRE_DIR)preprocessing $(DB_NAME) $(TRAINSET_PATH_BONDORA) $(TARGET_NAME_BONDORA) $(USELESS_ATTRIBUTES_BONDORA)  $(TRAIN_LABEL)

run_preprocess_test_bondora:
	$(PYTHON_INTERPRETER) $(PRE_DIR)preprocessing $(DB_NAME) $(TESTSET_PATH_BONDORA) $(TARGET_NAME_BONDORA) $(USELESS_ATTRIBUTES_BONDORA)  $(TEST_LABEL)

run_preprocess_bondora:
	$(PYTHON_INTERPRETER) $(PRE_DIR)preprocessing_of_state_of_art $(DB_NAME) $(DB_PATH_BONDORA) $(TARGET_NAME_BONDORA) $(USELESS_ATTRIBUTES_BONDORA)  $(SAMPLE_BONDORA) 

run_supervised_discretization_bondora:
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.supervised_discretization_process  $(DB_NAME)  $(TARGET_NAME_BONDORA)  "$(_PATH)"  $(NORMALIZATION_LABEL)  $(DATA_LABEL) $(SUB)

run_unsupervised_discretization_bondora:
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.unsupervised_discretization_process  $(DB_NAME)  $(TARGET_NAME_BONDORA)  "$(_PATH)"  $(NORMALIZATION_LABEL)  $(DATA_LABEL) 

	
########################################################################### GRAPH MANAGEMENT COMMANDS #########################################################################

# BIP GRAPH	
run_graph_modeling_bip_bondora:
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.graph.bip.build_graph $(DB_NAME) $(TARGET_NAME_BONDORA)  $(GRAPH_TYPE) $(DISCRETIZATION_TYPE) "$(TRAIN_PATH)" "$(TEST_PATH)"  "$(_DIR)" $(SUB)

run_compute_descriptors_bip_bondora: 
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.graph.bip.compute_descriptors $(TARGET_NAME_BONDORA)  $(BD_NAME) $(GRAPH_TYPE) $(ALPHA)  $(DISCRETIZATION_TYPE) "$(TRAIN_PATH)" "$(TEST_PATH)"  "$(_DIR)"  "$(GRAPH_DIR)" $(SUB)


# MOD GRAPH	 
run_graph_modeling_mod_bondora:
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.graph.mod.build_graph $(DB_NAME) $(TARGET_NAME_BONDORA)  $(GRAPH_TYPE) $(DISCRETIZATION_TYPE) "$(TRAIN_PATH)" "$(TEST_PATH)"  "$(_DIR)" $(SUB)

run_compute_descriptors_mod_bondora: 
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.graph.mod.compute_descriptors $(TARGET_NAME_BONDORA)  $(BD_NAME) $(GRAPH_TYPE) $(ALPHA)  $(DISCRETIZATION_TYPE) "$(TRAIN_PATH)" "$(TEST_PATH)"  "$(_DIR)"  "$(GRAPH_DIR)" $(SUB)



# COMPLETE GRAPHS MANAGEMENT
run_compute_descriptors_liu_v1_bondora: 
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.graph.liu.compute_descriptors_v1 $(TARGET_NAME_BONDORA)  $(BD_NAME) $(GRAPH_TYPE) "$(TRAIN_PATH)"  "$(TEST_PATH)"  "$(_DIR)"  "$(GRAPH_DIR)" $(SUB)

run_compute_descriptors_liu_v2_bondora: 
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.graph.liu.compute_descriptors_v2 $(TARGET_NAME_BONDORA)  $(BD_NAME) $(GRAPH_TYPE) "$(TRAIN_PATH)"  "$(TEST_PATH)"  "$(_DIR)"  "$(GRAPH_DIR)" $(SUB)

run_compute_descriptors_gui_bondora: 
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.graph.gui.compute_descriptors $(TARGET_NAME_BONDORA)  $(BD_NAME) $(GRAPH_TYPE) "$(TRAIN_PATH)"  "$(TEST_PATH)"  "$(_DIR)"  "$(GRAPH_DIR)" $(SUB)

run_graph_modeling_gui_bondora:
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.graph.gui.build_graph $(DB_NAME) $(TARGET_NAME_BONDORA)  $(GRAPH_TYPE) $(DISCRETIZATION_TYPE) "$(TRAIN_PATH)" "$(TEST_PATH)"  "$(_DIR)" $(SUB)

run_graph_modeling_liu_bondora:
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.graph.liu.build_graph $(DB_NAME) $(TARGET_NAME_BONDORA)  $(GRAPH_TYPE) $(DISCRETIZATION_TYPE) "$(TRAIN_PATH)" "$(TEST_PATH)"  "$(_DIR)" $(SUB)	

# GLO GRAPH
run_compute_descriptors_glo_bondora:
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.graph.complete.glo.compute_descriptors  $(TARGET_NAME_BONDORA)  $(BD_NAME) $(GRAPH_TYPE) $(ALPHA)	

run_graph_glo_bondora:
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.graph.complete.glo.launch  $(DB_NAME)	

# LOAN GRAPH	
run_graph_modeling_loan_bondora:
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.graph.loan.build_graph $(DB_NAME) $(TARGET_NAME_BONDORA) $(GRAPH_TYPE) "$(TRAIN_PATH)"  "$(TEST_PATH)"  "$(_DIR)" $(SUB)

run_compute_descriptors_loan_bondora: 
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.graph.loan.compute_descriptors $(TARGET_NAME_BONDORA)  $(BD_NAME)  $(GRAPH_TYPE)  $(ALPHA)  "$(TRAIN_PATH)"  "$(TEST_PATH)"  "$(GRAPH_DIR)"  "$(_DIR)" $(SUB)

###############################################################################################################################################################

run_make_configurations_bondora:
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.build_configurations $(TARGET_NAME_BONDORA) $(DB_NAME) $(SAVE_DIR) $(DISCRETIZATION_TYPE) $(GRAPH_TYPE) $(CLASSIC_TRAIN_PATH) $(NEW_DESCRIPTOR_TRAIN_PATH) $(SUB)

run_make_configurations_with_stepwise_bondora:
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.build_configurations_with_stepwise $(TARGET_NAME_BONDORA) $(DB_NAME) $(SAVE_DIR) $(DISCRETIZATION_TYPE) $(GRAPH_TYPE) $(CLASSIC_TRAIN_PATH) $(NEW_DESCRIPTOR_TRAIN_PATH) $(SUB)

run_select_features_bondora:
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.select_features_with_pvalue $(TARGET_NAME_BONDORA) $(DB_NAME) "$(TRAIN_PATH)" "$(SAVE_DIR)" $(SUB)
	
run_make_predictions_bondora:
	$(PYTHON_INTERPRETER) $(SRC_DIR)models.make_predictions_main $(TARGET_NAME_BONDORA) $(DB_NAME)  "$(TRAIN_PATH)"  "$(TEST_PATH)"  $(DISCRETIZATION_TYPE) $(GRAPH_TYPE) "$(CONFIG_PATH)" "$(SAVE_DIR)"  "$(CLASSIC_TRAIN_PATH)"  "$(CLASSIC_TEST_PATH)" "$(CLASSIC_CONFIG_PATH)" $(ALPHA) 

run_print_bondora:
	$(PYTHON_INTERPRETER) $(SRC_DIR)visualization.print_result  $(DB_NAME)	"$(_DIR)"  $(DISCRETIZATION_TYPE) $(GRAPH_TYPE) $(SUB)

run_plot_bondora:
	$(PYTHON_INTERPRETER) $(SRC_DIR)build_shap_plot.py $(TARGET_NAME_BONDORA)  $(DB_NAME)  "$(TRAIN_DESCRIPTORS_PATHS)" "$(TEST_DESCRIPTORS_PATHS)"  $(MODEL) $(DISCRETIZATION_TYPE) $(SUB)

run_summarize_bondora:
	$(PYTHON_INTERPRETER) $(SRC_DIR)summarize_shap_attributes_importance.py $(DB_NAME) $(DISCRETIZATION_TYPE) 

run_without_normalization_bondora:
	$(PYTHON_INTERPRETER) $(SRC_DIR)launchers.without_normalization.launch_per_database $(DB_NAME)

run_with_normalization_bondora:
	$(PYTHON_INTERPRETER) $(SRC_DIR)launchers.with_normalization.launch_per_database  $(DB_NAME)




#LENDING_CLUB

########################################################### DATA PREPROCESSING COMMANDS #######################################################################################

run_splitting_lending_club:
	$(PYTHON_INTERPRETER) $(PRE_DIR)splitting $(DB_NAME) $(DB_PATH_LENDING_CLUB)	$(TARGET_NAME_LENDING_CLUB) $(USELESS_ATTRIBUTES_LENDING_CLUB)  $(TARGET_VALUES_LENDING_CLUB)

run_engine_building_pre_lending_club:
	$(PYTHON_INTERPRETER) $(PRE_DIR)build_preprocess_engine $(DB_NAME) $(TRAINSET_PATH_LENDING_CLUB) $(TARGET_NAME_LENDING_CLUB) 

run_engine_building_unsupervised_discretization_lending_club:
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.engine.build_unsupervised_discretization_engine $(DB_NAME) $(TARGET_NAME_LENDING_CLUB) "$(_PATH)"  "$(_DIR)" 

run_engine_building_supervised_discretization_lending_club:
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.engine.build_supervised_discretization_engine $(DB_NAME) $(TARGET_NAME_LENDING_CLUB) "$(_PATH)" "$(_DIR)" $(SUB)

run_preprocess_train_lending_club:
	$(PYTHON_INTERPRETER) $(PRE_DIR)preprocessing $(DB_NAME) $(TRAINSET_PATH_LENDING_CLUB) $(TARGET_NAME_LENDING_CLUB) $(USELESS_ATTRIBUTES_LENDING_CLUB)  $(TRAIN_LABEL)

run_preprocess_test_lending_club:
	$(PYTHON_INTERPRETER) $(PRE_DIR)preprocessing $(DB_NAME) $(TESTSET_PATH_LENDING_CLUB) $(TARGET_NAME_LENDING_CLUB) $(USELESS_ATTRIBUTES_LENDING_CLUB)  $(TEST_LABEL)

run_preprocess_lending_club:
	$(PYTHON_INTERPRETER) $(PRE_DIR)preprocessing_of_state_of_art $(DB_NAME) $(DB_PATH_LENDING_CLUB) $(TARGET_NAME_LENDING_CLUB) $(USELESS_ATTRIBUTES_LENDING_CLUB)  $(SAMPLE_LENDING_CLUB) 

run_supervised_discretization_lending_club:
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.supervised_discretization_process  $(DB_NAME)  $(TARGET_NAME_LENDING_CLUB)  "$(_PATH)"  $(NORMALIZATION_LABEL)  $(DATA_LABEL) $(SUB)

run_unsupervised_discretization_lending_club:
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.unsupervised_discretization_process  $(DB_NAME)  $(TARGET_NAME_LENDING_CLUB)  "$(_PATH)"  $(NORMALIZATION_LABEL)  $(DATA_LABEL) 

	
########################################################################### GRAPH MANAGEMENT COMMANDS #########################################################################

# BIP GRAPH	
run_graph_modeling_bip_lending_club:
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.graph.bip.build_graph $(DB_NAME) $(TARGET_NAME_LENDING_CLUB)  $(GRAPH_TYPE) $(DISCRETIZATION_TYPE) "$(TRAIN_PATH)" "$(TEST_PATH)"  "$(_DIR)" $(SUB)

run_compute_descriptors_bip_lending_club: 
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.graph.bip.compute_descriptors $(TARGET_NAME_LENDING_CLUB)  $(BD_NAME) $(GRAPH_TYPE) $(ALPHA)  $(DISCRETIZATION_TYPE) "$(TRAIN_PATH)" "$(TEST_PATH)"  "$(_DIR)"  "$(GRAPH_DIR)" $(SUB)


# MOD GRAPH	 
run_graph_modeling_mod_lending_club:
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.graph.mod.build_graph $(DB_NAME) $(TARGET_NAME_LENDING_CLUB)  $(GRAPH_TYPE) $(DISCRETIZATION_TYPE) "$(TRAIN_PATH)" "$(TEST_PATH)"  "$(_DIR)" $(SUB)

run_compute_descriptors_mod_lending_club: 
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.graph.mod.compute_descriptors $(TARGET_NAME_LENDING_CLUB)  $(BD_NAME) $(GRAPH_TYPE) $(ALPHA)  $(DISCRETIZATION_TYPE) "$(TRAIN_PATH)" "$(TEST_PATH)"  "$(_DIR)"  "$(GRAPH_DIR)" $(SUB)



# COMPLETE GRAPHS MANAGEMENT
run_compute_descriptors_liu_v1_lending_club: 
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.graph.liu.compute_descriptors_v1 $(TARGET_NAME_LENDING_CLUB)  $(BD_NAME) $(GRAPH_TYPE) "$(TRAIN_PATH)"  "$(TEST_PATH)"  "$(_DIR)"  "$(GRAPH_DIR)" $(SUB)

run_compute_descriptors_liu_v2_lending_club: 
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.graph.liu.compute_descriptors_v2 $(TARGET_NAME_LENDING_CLUB)  $(BD_NAME) $(GRAPH_TYPE) "$(TRAIN_PATH)"  "$(TEST_PATH)"  "$(_DIR)"  "$(GRAPH_DIR)" $(SUB)

run_compute_descriptors_gui_lending_club: 
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.graph.gui.compute_descriptors $(TARGET_NAME_LENDING_CLUB)  $(BD_NAME) $(GRAPH_TYPE) "$(TRAIN_PATH)"  "$(TEST_PATH)"  "$(_DIR)"  "$(GRAPH_DIR)" $(SUB)

run_graph_modeling_gui_lending_club:
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.graph.gui.build_graph $(DB_NAME) $(TARGET_NAME_LENDING_CLUB)  $(GRAPH_TYPE) $(DISCRETIZATION_TYPE) "$(TRAIN_PATH)" "$(TEST_PATH)"  "$(_DIR)" $(SUB)

run_graph_modeling_liu_lending_club:
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.graph.liu.build_graph $(DB_NAME) $(TARGET_NAME_LENDING_CLUB)  $(GRAPH_TYPE) $(DISCRETIZATION_TYPE) "$(TRAIN_PATH)" "$(TEST_PATH)"  "$(_DIR)" $(SUB)	

# GLO GRAPH
run_compute_descriptors_glo_lending_club:
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.graph.complete.glo.compute_descriptors  $(TARGET_NAME_LENDING_CLUB)  $(BD_NAME) $(GRAPH_TYPE) $(ALPHA)	

run_graph_glo_lending_club:
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.graph.complete.glo.launch  $(DB_NAME)	

# LOAN GRAPH	
run_graph_modeling_loan_lending_club:
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.graph.loan.build_graph $(DB_NAME) $(TARGET_NAME_LENDING_CLUB) $(GRAPH_TYPE) "$(TRAIN_PATH)"  "$(TEST_PATH)"  "$(_DIR)" $(SUB)

run_compute_descriptors_loan_lending_club: 
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.graph.loan.compute_descriptors $(TARGET_NAME_LENDING_CLUB)  $(BD_NAME)  $(GRAPH_TYPE)  $(ALPHA)  "$(TRAIN_PATH)"  "$(TEST_PATH)"  "$(GRAPH_DIR)"  "$(_DIR)" $(SUB)

###############################################################################################################################################################

run_make_configurations_lending_club:
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.build_configurations $(TARGET_NAME_LENDING_CLUB) $(DB_NAME) $(SAVE_DIR) $(DISCRETIZATION_TYPE) $(GRAPH_TYPE) $(CLASSIC_TRAIN_PATH) $(NEW_DESCRIPTOR_TRAIN_PATH) $(SUB)

run_make_configurations_with_stepwise_lending_club:
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.build_configurations_with_stepwise $(TARGET_NAME_LENDING_CLUB) $(DB_NAME) $(SAVE_DIR) $(DISCRETIZATION_TYPE) $(GRAPH_TYPE) $(CLASSIC_TRAIN_PATH) $(NEW_DESCRIPTOR_TRAIN_PATH) $(SUB)

run_select_features_lending_club:
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.select_features_with_pvalue $(TARGET_NAME_LENDING_CLUB) $(DB_NAME) "$(TRAIN_PATH)" "$(SAVE_DIR)" $(SUB)
	
run_make_predictions_lending_club:
	$(PYTHON_INTERPRETER) $(SRC_DIR)models.make_predictions_main $(TARGET_NAME_LENDING_CLUB) $(DB_NAME)  "$(TRAIN_PATH)"  "$(TEST_PATH)"  $(DISCRETIZATION_TYPE) $(GRAPH_TYPE) "$(CONFIG_PATH)" "$(SAVE_DIR)"  "$(CLASSIC_TRAIN_PATH)"  "$(CLASSIC_TEST_PATH)" "$(CLASSIC_CONFIG_PATH)" $(ALPHA) 

run_print_lending_club:
	$(PYTHON_INTERPRETER) $(SRC_DIR)visualization.print_result  $(DB_NAME)	"$(_DIR)"  $(DISCRETIZATION_TYPE) $(GRAPH_TYPE) $(SUB)

run_plot_lending_club:
	$(PYTHON_INTERPRETER) $(SRC_DIR)build_shap_plot.py $(TARGET_NAME_LENDING_CLUB)  $(DB_NAME)  "$(TRAIN_DESCRIPTORS_PATHS)" "$(TEST_DESCRIPTORS_PATHS)"  $(MODEL) $(DISCRETIZATION_TYPE) $(SUB)

run_summarize_lending_club:
	$(PYTHON_INTERPRETER) $(SRC_DIR)summarize_shap_attributes_importance.py $(DB_NAME) $(DISCRETIZATION_TYPE) 

run_without_normalization_lending_club:
	$(PYTHON_INTERPRETER) $(SRC_DIR)launchers.without_normalization.launch_per_database $(DB_NAME)

run_with_normalization_lending_club:
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
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.engine.build_supervised_discretization_engine $(DB_NAME) $(TARGET_NAME_PROSPER) "$(_PATH)" "$(_DIR)" $(SUB)

run_preprocess_train_prosper:
	$(PYTHON_INTERPRETER) $(PRE_DIR)preprocessing $(DB_NAME) $(TRAINSET_PATH_PROSPER) $(TARGET_NAME_PROSPER) $(USELESS_ATTRIBUTES_PROSPER)  $(TRAIN_LABEL)

run_preprocess_test_prosper:
	$(PYTHON_INTERPRETER) $(PRE_DIR)preprocessing $(DB_NAME) $(TESTSET_PATH_PROSPER) $(TARGET_NAME_PROSPER) $(USELESS_ATTRIBUTES_PROSPER)  $(TEST_LABEL)

run_preprocess_prosper:
	$(PYTHON_INTERPRETER) $(PRE_DIR)preprocessing_of_state_of_art $(DB_NAME) $(DB_PATH_PROSPER) $(TARGET_NAME_PROSPER) $(USELESS_ATTRIBUTES_PROSPER)  $(SAMPLE_PROSPER) 

run_supervised_discretization_prosper:
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.supervised_discretization_process  $(DB_NAME)  $(TARGET_NAME_PROSPER)  "$(_PATH)"  $(NORMALIZATION_LABEL)  $(DATA_LABEL) $(SUB)

run_unsupervised_discretization_prosper:
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.unsupervised_discretization_process  $(DB_NAME)  $(TARGET_NAME_PROSPER)  "$(_PATH)"  $(NORMALIZATION_LABEL)  $(DATA_LABEL) 

	
########################################################################### GRAPH MANAGEMENT COMMANDS #########################################################################

# BIP GRAPH	
run_graph_modeling_bip_prosper:
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.graph.bip.build_graph $(DB_NAME) $(TARGET_NAME_PROSPER)  $(GRAPH_TYPE) $(DISCRETIZATION_TYPE) "$(TRAIN_PATH)" "$(TEST_PATH)"  "$(_DIR)" $(SUB)

run_compute_descriptors_bip_prosper: 
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.graph.bip.compute_descriptors $(TARGET_NAME_PROSPER)  $(BD_NAME) $(GRAPH_TYPE) $(ALPHA)  $(DISCRETIZATION_TYPE) "$(TRAIN_PATH)" "$(TEST_PATH)"  "$(_DIR)"  "$(GRAPH_DIR)" $(SUB)


# MOD GRAPH	 
run_graph_modeling_mod_prosper:
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.graph.mod.build_graph $(DB_NAME) $(TARGET_NAME_PROSPER)  $(GRAPH_TYPE) $(DISCRETIZATION_TYPE) "$(TRAIN_PATH)" "$(TEST_PATH)"  "$(_DIR)" $(SUB)

run_compute_descriptors_mod_prosper: 
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.graph.mod.compute_descriptors $(TARGET_NAME_PROSPER)  $(BD_NAME) $(GRAPH_TYPE) $(ALPHA)  $(DISCRETIZATION_TYPE) "$(TRAIN_PATH)" "$(TEST_PATH)"  "$(_DIR)"  "$(GRAPH_DIR)" $(SUB)



# COMPLETE GRAPHS MANAGEMENT
run_compute_descriptors_liu_v1_prosper: 
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.graph.liu.compute_descriptors_v1 $(TARGET_NAME_PROSPER)  $(BD_NAME) $(GRAPH_TYPE) "$(TRAIN_PATH)"  "$(TEST_PATH)"  "$(_DIR)"  "$(GRAPH_DIR)" $(SUB)

run_compute_descriptors_liu_v2_prosper: 
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.graph.liu.compute_descriptors_v2 $(TARGET_NAME_PROSPER)  $(BD_NAME) $(GRAPH_TYPE) "$(TRAIN_PATH)"  "$(TEST_PATH)"  "$(_DIR)"  "$(GRAPH_DIR)" $(SUB)

run_compute_descriptors_gui_prosper: 
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.graph.gui.compute_descriptors $(TARGET_NAME_PROSPER)  $(BD_NAME) $(GRAPH_TYPE) "$(TRAIN_PATH)"  "$(TEST_PATH)"  "$(_DIR)"  "$(GRAPH_DIR)" $(SUB)

run_graph_modeling_gui_prosper:
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.graph.gui.build_graph $(DB_NAME) $(TARGET_NAME_PROSPER)  $(GRAPH_TYPE) $(DISCRETIZATION_TYPE) "$(TRAIN_PATH)" "$(TEST_PATH)"  "$(_DIR)" $(SUB)

run_graph_modeling_liu_prosper:
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.graph.liu.build_graph $(DB_NAME) $(TARGET_NAME_PROSPER)  $(GRAPH_TYPE) $(DISCRETIZATION_TYPE) "$(TRAIN_PATH)" "$(TEST_PATH)"  "$(_DIR)" $(SUB)	

# GLO GRAPH
run_compute_descriptors_glo_prosper:
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.graph.complete.glo.compute_descriptors  $(TARGET_NAME_PROSPER)  $(BD_NAME) $(GRAPH_TYPE) $(ALPHA)	

run_graph_glo_prosper:
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.graph.complete.glo.launch  $(DB_NAME)	

# LOAN GRAPH	
run_graph_modeling_loan_prosper:
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.graph.loan.build_graph $(DB_NAME) $(TARGET_NAME_PROSPER) $(GRAPH_TYPE) "$(TRAIN_PATH)"  "$(TEST_PATH)"  "$(_DIR)" $(SUB)

run_compute_descriptors_loan_prosper: 
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.graph.loan.compute_descriptors $(TARGET_NAME_PROSPER)  $(BD_NAME)  $(GRAPH_TYPE)  $(ALPHA)  "$(TRAIN_PATH)"  "$(TEST_PATH)"  "$(GRAPH_DIR)"  "$(_DIR)" $(SUB)

###############################################################################################################################################################

run_make_configurations_prosper:
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.build_configurations $(TARGET_NAME_PROSPER) $(DB_NAME) $(SAVE_DIR) $(DISCRETIZATION_TYPE) $(GRAPH_TYPE) $(CLASSIC_TRAIN_PATH) $(NEW_DESCRIPTOR_TRAIN_PATH) $(SUB)

run_make_configurations_with_stepwise_prosper:
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.build_configurations_with_stepwise $(TARGET_NAME_PROSPER) $(DB_NAME) $(SAVE_DIR) $(DISCRETIZATION_TYPE) $(GRAPH_TYPE) $(CLASSIC_TRAIN_PATH) $(NEW_DESCRIPTOR_TRAIN_PATH) $(SUB)

run_select_features_prosper:
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.select_features_with_pvalue $(TARGET_NAME_PROSPER) $(DB_NAME) "$(TRAIN_PATH)" "$(SAVE_DIR)" $(SUB)
	
run_make_predictions_prosper:
	$(PYTHON_INTERPRETER) $(SRC_DIR)models.make_predictions_main $(TARGET_NAME_PROSPER) $(DB_NAME)  "$(TRAIN_PATH)"  "$(TEST_PATH)"  $(DISCRETIZATION_TYPE) $(GRAPH_TYPE) "$(CONFIG_PATH)" "$(SAVE_DIR)"  "$(CLASSIC_TRAIN_PATH)"  "$(CLASSIC_TEST_PATH)" "$(CLASSIC_CONFIG_PATH)" $(ALPHA) 

run_print_prosper:
	$(PYTHON_INTERPRETER) $(SRC_DIR)visualization.print_result  $(DB_NAME)	"$(_DIR)"  $(DISCRETIZATION_TYPE) $(GRAPH_TYPE) $(SUB)

run_plot_prosper:
	$(PYTHON_INTERPRETER) $(SRC_DIR)build_shap_plot.py $(TARGET_NAME_PROSPER)  $(DB_NAME)  "$(TRAIN_DESCRIPTORS_PATHS)" "$(TEST_DESCRIPTORS_PATHS)"  $(MODEL) $(DISCRETIZATION_TYPE) $(SUB)

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
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.engine.build_supervised_discretization_engine $(DB_NAME) $(TARGET_NAME_SME) "$(_PATH)" "$(_DIR)" $(SUB)

run_preprocess_train_sme:
	$(PYTHON_INTERPRETER) $(PRE_DIR)preprocessing $(DB_NAME) $(TRAINSET_PATH_SME) $(TARGET_NAME_SME) $(USELESS_ATTRIBUTES_SME)  $(TRAIN_LABEL)

run_preprocess_test_sme:
	$(PYTHON_INTERPRETER) $(PRE_DIR)preprocessing $(DB_NAME) $(TESTSET_PATH_SME) $(TARGET_NAME_SME) $(USELESS_ATTRIBUTES_SME)  $(TEST_LABEL)

run_preprocess_sme:
	$(PYTHON_INTERPRETER) $(PRE_DIR)preprocessing_of_state_of_art $(DB_NAME) $(DB_PATH_SME) $(TARGET_NAME_SME) $(USELESS_ATTRIBUTES_SME)  $(SAMPLE_SME) 

run_supervised_discretization_sme:
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.supervised_discretization_process  $(DB_NAME)  $(TARGET_NAME_SME)  "$(_PATH)"  $(NORMALIZATION_LABEL)  $(DATA_LABEL) $(SUB)

run_unsupervised_discretization_sme:
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.unsupervised_discretization_process  $(DB_NAME)  $(TARGET_NAME_SME)  "$(_PATH)"  $(NORMALIZATION_LABEL)  $(DATA_LABEL) 

	
########################################################################### GRAPH MANAGEMENT COMMANDS #########################################################################

# BIP GRAPH	
run_graph_modeling_bip_sme:
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.graph.bip.build_graph $(DB_NAME) $(TARGET_NAME_SME)  $(GRAPH_TYPE) $(DISCRETIZATION_TYPE) "$(TRAIN_PATH)" "$(TEST_PATH)"  "$(_DIR)" $(SUB)

run_compute_descriptors_bip_sme: 
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.graph.bip.compute_descriptors $(TARGET_NAME_SME)  $(BD_NAME) $(GRAPH_TYPE) $(ALPHA)  $(DISCRETIZATION_TYPE) "$(TRAIN_PATH)" "$(TEST_PATH)"  "$(_DIR)"  "$(GRAPH_DIR)" $(SUB)


# MOD GRAPH	 
run_graph_modeling_mod_sme:
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.graph.mod.build_graph $(DB_NAME) $(TARGET_NAME_SME)  $(GRAPH_TYPE) $(DISCRETIZATION_TYPE) "$(TRAIN_PATH)" "$(TEST_PATH)"  "$(_DIR)" $(SUB)

run_compute_descriptors_mod_sme: 
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.graph.mod.compute_descriptors $(TARGET_NAME_SME)  $(BD_NAME) $(GRAPH_TYPE) $(ALPHA)  $(DISCRETIZATION_TYPE) "$(TRAIN_PATH)" "$(TEST_PATH)"  "$(_DIR)"  "$(GRAPH_DIR)" $(SUB)



# COMPLETE GRAPHS MANAGEMENT
run_compute_descriptors_liu_v1_sme: 
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.graph.liu.compute_descriptors_v1 $(TARGET_NAME_SME)  $(BD_NAME) $(GRAPH_TYPE) "$(TRAIN_PATH)"  "$(TEST_PATH)"  "$(_DIR)"  "$(GRAPH_DIR)" $(SUB)

run_compute_descriptors_liu_v2_sme: 
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.graph.liu.compute_descriptors_v2 $(TARGET_NAME_SME)  $(BD_NAME) $(GRAPH_TYPE) "$(TRAIN_PATH)"  "$(TEST_PATH)"  "$(_DIR)"  "$(GRAPH_DIR)" $(SUB)

run_compute_descriptors_gui_sme: 
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.graph.gui.compute_descriptors $(TARGET_NAME_SME)  $(BD_NAME) $(GRAPH_TYPE) "$(TRAIN_PATH)"  "$(TEST_PATH)"  "$(_DIR)"  "$(GRAPH_DIR)" $(SUB)

run_graph_modeling_gui_sme:
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.graph.gui.build_graph $(DB_NAME) $(TARGET_NAME_SME)  $(GRAPH_TYPE) $(DISCRETIZATION_TYPE) "$(TRAIN_PATH)" "$(TEST_PATH)"  "$(_DIR)" $(SUB)

run_graph_modeling_liu_sme:
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.graph.liu.build_graph $(DB_NAME) $(TARGET_NAME_SME)  $(GRAPH_TYPE) $(DISCRETIZATION_TYPE) "$(TRAIN_PATH)" "$(TEST_PATH)"  "$(_DIR)" $(SUB)	

# GLO GRAPH
run_compute_descriptors_glo_sme:
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.graph.complete.glo.compute_descriptors  $(TARGET_NAME_SME)  $(BD_NAME) $(GRAPH_TYPE) $(ALPHA)	

run_graph_glo_sme:
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.graph.complete.glo.launch  $(DB_NAME)	

# LOAN GRAPH	
run_graph_modeling_loan_sme:
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.graph.loan.build_graph $(DB_NAME) $(TARGET_NAME_SME) $(GRAPH_TYPE) "$(TRAIN_PATH)"  "$(TEST_PATH)"  "$(_DIR)" $(SUB)

run_compute_descriptors_loan_sme: 
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.graph.loan.compute_descriptors $(TARGET_NAME_SME)  $(BD_NAME)  $(GRAPH_TYPE)  $(ALPHA)  "$(TRAIN_PATH)"  "$(TEST_PATH)"  "$(GRAPH_DIR)"  "$(_DIR)" $(SUB)

###############################################################################################################################################################

run_make_configurations_sme:
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.build_configurations $(TARGET_NAME_SME) $(DB_NAME) $(SAVE_DIR) $(DISCRETIZATION_TYPE) $(GRAPH_TYPE) $(CLASSIC_TRAIN_PATH) $(NEW_DESCRIPTOR_TRAIN_PATH) $(SUB)

run_make_configurations_with_stepwise_sme:
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.build_configurations_with_stepwise $(TARGET_NAME_SME) $(DB_NAME) $(SAVE_DIR) $(DISCRETIZATION_TYPE) $(GRAPH_TYPE) $(CLASSIC_TRAIN_PATH) $(NEW_DESCRIPTOR_TRAIN_PATH) $(SUB)

run_select_features_sme:
	$(PYTHON_INTERPRETER) $(SRC_DIR)features.select_features_with_pvalue $(TARGET_NAME_SME) $(DB_NAME) "$(TRAIN_PATH)" "$(SAVE_DIR)" $(SUB)
	
run_make_predictions_sme:
	$(PYTHON_INTERPRETER) $(SRC_DIR)models.make_predictions_main $(TARGET_NAME_SME) $(DB_NAME)  "$(TRAIN_PATH)"  "$(TEST_PATH)"  $(DISCRETIZATION_TYPE) $(GRAPH_TYPE) "$(CONFIG_PATH)" "$(SAVE_DIR)"  "$(CLASSIC_TRAIN_PATH)"  "$(CLASSIC_TEST_PATH)" "$(CLASSIC_CONFIG_PATH)" $(ALPHA) 

run_print_sme:
	$(PYTHON_INTERPRETER) $(SRC_DIR)visualization.print_result  $(DB_NAME)	"$(_DIR)"  $(DISCRETIZATION_TYPE) $(GRAPH_TYPE) $(SUB)

run_plot_sme:
	$(PYTHON_INTERPRETER) $(SRC_DIR)build_shap_plot.py $(TARGET_NAME_SME)  $(DB_NAME)  "$(TRAIN_DESCRIPTORS_PATHS)" "$(TEST_DESCRIPTORS_PATHS)"  $(MODEL) $(DISCRETIZATION_TYPE) $(SUB)

run_summarize_sme:
	$(PYTHON_INTERPRETER) $(SRC_DIR)summarize_shap_attributes_importance.py $(DB_NAME) $(DISCRETIZATION_TYPE) 

run_without_normalization_sme:
	$(PYTHON_INTERPRETER) $(SRC_DIR)launchers.without_normalization.launch_per_database $(DB_NAME)

run_with_normalization_sme:
	$(PYTHON_INTERPRETER) $(SRC_DIR)launchers.with_normalization.launch_per_database  $(DB_NAME) 











