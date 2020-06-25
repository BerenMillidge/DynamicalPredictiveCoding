import os
import sys
generated_name = str(sys.argv[1])
log_path = str(sys.argv[2])
save_path = str(sys.argv[3])
exp_name = str(sys.argv[4])
base_call = "python main.py"
output_file = open(generated_name, "w")
seeds = 5
condition="amortisation_fixed_predictions"
for s in range(seeds):
    lpath = log_path + "/"+str(exp_name) +"_"+condition + "/" + str(s)
    spath = save_path + "/" + str(exp_name) +"_"+condition + "/" + str(s)
    final_call = base_call + " --logdir " + str(lpath) + " --savedir " + str(spath) + "--with_amortisation True --fixed_predictions True"
    print(final_call)
    print(final_call, file=output_file)

condition="amortisation_no_fixed_predictions"
for s in range(seeds):
    lpath = log_path + "/"+str(exp_name) +"_"+condition + "/" + str(s)
    spath = save_path + "/" + str(exp_name) +"_"+condition + "/" + str(s)
    final_call = base_call + " --logdir " + str(lpath) + " --savedir " + str(spath) + "--with_amortisation True --fixed_predictions False"
    print(final_call)
    print(final_call, file=output_file)

condition="no_amortisation_fixed_predictions"
for s in range(seeds):
    lpath = log_path + "/"+str(exp_name) +"_"+condition + "/" + str(s)
    spath = save_path + "/" + str(exp_name) +"_"+condition + "/" + str(s)
    final_call = base_call + " --logdir " + str(lpath) + " --savedir " + str(spath) + "--with_amortisation False --fixed_predictions True"
    print(final_call)
    print(final_call, file=output_file)

condition="no_amortisation_no_fixed_predictions"
for s in range(seeds):
    lpath = log_path + "/"+str(exp_name) +"_"+condition + "/" + str(s)
    spath = save_path + "/" + str(exp_name) +"_"+condition + "/" + str(s)
    final_call = base_call + " --logdir " + str(lpath) + " --savedir " + str(spath) + "--with_amortisation False --fixed_predictions False"
    print(final_call)
    print(final_call, file=output_file)




condition="dynamical_weihgts_amortisation_fixed_predictions"
for s in range(seeds):
    lpath = log_path + "/"+str(exp_name) +"_"+condition + "/" + str(s)
    spath = save_path + "/" + str(exp_name) +"_"+condition + "/" + str(s)
    final_call = base_call + " --logdir " + str(lpath) + " --savedir " + str(spath) + "--with_amortisation True --fixed_predictions True --continual_weight_update True"
    print(final_call)
    print(final_call, file=output_file)

condition="dynamical_weights_amortisation_no_fixed_predictions"
for s in range(seeds):
    lpath = log_path + "/"+str(exp_name) +"_"+condition + "/" + str(s)
    spath = save_path + "/" + str(exp_name) +"_"+condition + "/" + str(s)
    final_call = base_call + " --logdir " + str(lpath) + " --savedir " + str(spath) + "--with_amortisation True --fixed_predictions False --continual_weight_update True"
    print(final_call)
    print(final_call, file=output_file)

condition="dynamical_weightsno_amortisation_fixed_predictions"
for s in range(seeds):
    lpath = log_path + "/"+str(exp_name) +"_"+condition + "/" + str(s)
    spath = save_path + "/" + str(exp_name) +"_"+condition + "/" + str(s)
    final_call = base_call + " --logdir " + str(lpath) + " --savedir " + str(spath) + "--with_amortisation False --fixed_predictions True --continual_weight_update True"
    print(final_call)
    print(final_call, file=output_file)

condition="dynamical_weights_no_amortisation_no_fixed_predictions"
for s in range(seeds):
    lpath = log_path + "/"+str(exp_name) +"_"+condition + "/" + str(s)
    spath = save_path + "/" + str(exp_name) +"_"+condition + "/" + str(s)
    final_call = base_call + " --logdir " + str(lpath) + " --savedir " + str(spath) + "--with_amortisation False --fixed_predictions False --continual_weight_update True"
    print(final_call)
    print(final_call, file=output_file)