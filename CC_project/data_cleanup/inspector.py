import os, random, time, shutil
from datetime import datetime
direc = '/n/eddy_lab/users/npcarter/null-model/sequence_generation/parsed_sockets_output/has_cc/'

to100 = 0
to400 = 0
to_max = 0
def length_partition(filename, length, base_dst):
    global to100, to400, to_max
    if (length <= 100): 
        shutil.copy(os.path.join(direc, filename), os.path.join(base_dst, 'to100/')) 
        to100 = to100 + 1
    elif (length <= 400): 
        shutil.copy(os.path.join(direc, filename), os.path.join(base_dst, 'to400/'))
        to400 = to400 + 1
    else: 
        shutil.copy(os.path.join(direc, filename), os.path.join(base_dst, 'to_max/'))
        to_max = to_max + 1


now = datetime.now()
t0 = time.perf_counter()
histog = {}
num_files = 0
for filename in os.listdir(direc):
# for i in range(5):
    num_files = num_files + 1
    # filename = random.choice(os.listdir(direc))
    # print(fil)
    with open(os.path.join(direc, filename), 'r') as f:
        contents = f.readlines()
        # print(contents)   
        coiled_coils = contents[2][:-1] # to remove the \n. NOTE: there is an extra - at the end, but we will keep it for now to be used as extra buffer character in the for loop
        # print(coiled_coils)
        # print(contents[0][:-2])   
        found_cc = False
        length = 0
        for char in coiled_coils:
            if (char != '-'): # this residue is part of a coiled coil
                length = length + 1 
            else: # this is NOT a coiled coil
                if (length != 0): # coiled coil just ended
                    if length not in histog: # doesn't have length yet, so create it    
                        histog[length] = 1
                    else: # update histogram frequency value for this length
                        freq = histog[length]
                        histog[length] = freq + 1
                    length = 0
                    
                    if (not found_cc): # this is first cc found  
                        found_cc = True
                        length_partition(filename, len(coiled_coils) - 1, '/n/eddy_lab/users/rangerkuang/CC_data/has_cc_real/')  
        if (not found_cc): # went through entire protein, and didn't find a CC
            length_partition(filename, len(coiled_coils) - 1, '/n/eddy_lab/users/rangerkuang/CC_data/no_cc_real/')

print(histog)
t1 = time.perf_counter()
time_elapsed = t1 - t0

dt_string = now.strftime("%m/%d/%Y %H:%M:%S")

histog_data = open('/n/eddy_lab/users/rangerkuang/CC_project/histog_data', 'a')
histog_data.write(f"[{dt_string}] time elapsed: {time_elapsed} in {num_files} files | with split: {to100}:{to400}:{to_max} " + repr(histog) + "\n")
histog_data.close()
