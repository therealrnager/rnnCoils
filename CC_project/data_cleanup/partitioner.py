import os, shutil, random 
def partition():
    base = os.path.join('/n/eddy_lab/users/rangerkuang/CC_data/')
    # FULL SUBDIR: subdir = ['has_cc_real/to100/', 'has_cc_real/to400/', 'has_cc_real/to_max/', 'no_cc_real/to100/', 'no_cc_real/to400/', 'no_cc_real/to_max/']
    # subdir = ['has_cc_real/to100/']
    subdir = ['has_cc_real/to400/', 'has_cc_real/to_max/', 'no_cc_real/to100/', 'no_cc_real/to400/', 'no_cc_real/to_max/']
    # FULL TEN PERCENT LIST : ten_percent_list = [115, 574, 339, 3205, 7615, 2053]
    ten_percent_list = [574, 339, 3205, 7615, 2053]
    for x in range(len(subdir)):
        dir = subdir[x]
        direc = os.path.join(os.path.join(base, dir), 'tmp')
        ten_percent = ten_percent_list[x]

        seq_list = os.listdir(direc)
        random.shuffle(seq_list)
        for i in range(0, ten_percent):
            path = os.path.join(direc, seq_list[i])
            shutil.move(path, os.path.join(os.path.join(base, dir), 'valid'))
        print(direc, " valid done")
        for i in range(ten_percent, 2 * ten_percent):
            path = os.path.join(direc, seq_list[i])
            shutil.move(path, os.path.join(os.path.join(base, dir), 'test'))
        print(direc, " test done")
        for i in range(2 * ten_percent, len(seq_list)):
            path = os.path.join(direc, seq_list[i])
            shutil.move(path, os.path.join(os.path.join(base, dir), 'train'))
        print(direc, " train done")

        # i = 0
        # while (i < ten_percent):
            
        #     #filename = random.choice()
        #     path = os.path.join(direc, filename)
        #     if os.path.isdir(path):
        #     # skip directories
        #         continue
        #     i = i + 1
        #     shutil.move(path, os.path.join(base, 'valid/'))
        # print(folder, ' valid done')
        # i = 0
        # while (i < ten_percent):
        #     filename = random.choice(os.listdir(base))
        #     path = os.path.join(base, filename)
        #     if os.path.isdir(path):
        #     # skip directories
        #         continue
        #     i = i + 1
        #     shutil.move(path, os.path.join(base, 'test/'))
        # print(folder, ' test done')
        # for entry in os.scandir(base):
        #     if entry.is_dir():
        #         continue   
        #     shutil.move(entry.path, os.path.join(base, 'train/'))
        # print(folder, ' train done')


partition()
#partition('no_cc_real/to_max/', 9114)
