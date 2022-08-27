import os

base_direc = '/n/eddy_lab/users/rangerkuang/CC_data/'

seq_histogram = {}
cc_histogram = {}

def length_parse(end):
    global base_direc, seq_histogram, cc_histogram
    # subdir = ['has_cc_real/to100/', 'has_cc_real/to400/', 'has_cc_real/to_max/', 'no_cc_real/to100/', 'no_cc_real/to400/', 'no_cc_real/to_max/']
    subdir = ['has_cc_real/to_max/', 'has_cc_real/to400/']
    # subdir = ['has_cc_real/to100/', 'has_cc_real/to_max/']

    for i in subdir:
        mid_direc = os.path.join(base_direc, i)
        direc = os.path.join(mid_direc, end)
        for filename in os.scandir(direc):
            with open(filename.path, 'r') as f:
                contents = f.readlines()


                
            
                coiled_coils = contents[2][:-1] # to remove the \n. NOTE: there is an extra - at the end, but we will keep it for now to be used as extra buffer character in the for loop

                seqlength = len(coiled_coils) - 1
                # if (seqlength > 1000):
                #     print("seqlength over 1600, ", filename.path)

                # interpret sequence length
                if seqlength not in seq_histogram: # doesn't have length yet, so create it    
                    seq_histogram[seqlength] = 1
                else: # update histogram frequency value for this length
                    freq = seq_histogram[seqlength]
                    seq_histogram[seqlength] = freq + 1

                # interpret coiled coil lengths
                length = 0
                for char in coiled_coils:
                    if (char != '-'): # this residue is part of a coiled coil
                        length = length + 1 
                    else: # this is NOT a coiled coil
                        if (length != 0): # coiled coil just ended
                            if (length > 30 and length < 60):
                                print(filename.path)
                            if length not in cc_histogram: # doesn't have length yet, so create it    
                                cc_histogram[length] = 1
                            else: # update histogram frequency value for this length
                                freq = cc_histogram[length]
                                cc_histogram[length] = freq + 1
                            length = 0


                            
                        
        print(i, ' is done!')


length_parse('test')

# length_parse('valid')
# length_parse('train')
# length_parse('test')


histog_data = open('/n/eddy_lab/users/rangerkuang/CC_project/data_cleanup/histog_data', 'a')
histog_data.write("seq histogram: " + repr(seq_histogram) + "\n")
histog_data.write("CC histogram: " + repr(cc_histogram) + "\n")
histog_data.close()

#print(seq_histogram)
#print(cc_histogram)

