############################################################################
#
# Copyright (c) 2016 ICT MCG Group, Inc. All Rights Reserved
#
###########################################################################
"""
Brief:

Authors: zhouxing(@ict.ac.cn)
"""
import sys
import os
def add_pattern(pattern, dic):
    dic.setdefault(pattern, 0)
    dic[pattern] += 1
def main():
    data_dir = sys.argv[1]
    category = sys.argv[2]
    pattern_cnt = {}
    for one_file in os.listdir(data_dir):
        f = open(data_dir + "/" + one_file)
        while 1:
            line = f.readline()
            if not line:
                break
            arr = line.rstrip().split('\t')
            if arr[0] != category:
                for i in range(6):
                    line = f.readline()
                continue
            # next 6 line are patterns
            line = f.readline()
            line = f.readline().rstrip()
            add_pattern(line, pattern_cnt)

            line = f.readline()
            line = f.readline().rstrip()
            #add_pattern(line, pattern_cnt)
            for i in range(2):
                line = f.readline()
    sorted_list= sorted(pattern_cnt.iteritems(), key=lambda d:d[1], reverse = True)
    for key, val in sorted_list:
        print key, val

if __name__ == '__main__':
    main()
# vim: set expandtab ts=4 sw=4 sts=4 tw=100:
