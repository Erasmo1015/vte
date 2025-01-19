# Step 1 : read from experts/HalfCheetah-v3_8.pkl and experts/HalfCheetah-v3_9.pkl
# Step 2: combine the two files into one file
# Step 3: save it in experrts/ HalfCheetah-v3_Gemini.pkl

# Step 1 : read from experts/HalfCheetah-v3_8.pkl and experts/HalfCheetah-v3_9.pkl
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

def list_of_strings(arg):
    return arg.split(',')

def main():
    # set up argparser and 
    # read args of data1 and data2 path:
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("data_list", type=list_of_strings, help="List of expert demos to combine")
    # arg_parser.add_argument('data1', type=str, default='experts/HalfCheetah-v3_8.pkl')
    # arg_parser.add_argument('data2', type=str, default='experts/HalfCheetah-v3_6_2327r.pkl')
    arg_parser.add_argument('segment_len', type=int, default=5) # segment legnth for each file
    arg_parser.add_argument('save_path', type=str, default='experts/HalfCheetah-v3_Gemini_2k+6k.pkl')
    args = arg_parser.parse_args()

    data = []
    for i in range(len(args.data_list)):
        with open(args.data_list[i], 'rb') as f:
            data.append(pickle.load(f))
    # with open(args.data1, 'rb') as f:
    #     data1 = pickle.load(f)
    # with open(args.data2, 'rb') as f:
    #     data2 = pickle.load(f)

    # Step 2: combine the two files into one file
    # for each file, take three items out of each value list and combine them into a new list
    def combine_value(data):
        new_data = {}
        for key in data[0].keys():
            for i in range(len(data)):
                if key not in new_data:
                    new_data[key] = []
                new_data[key].extend(data[i][key][:args.segment_len])
                ## --> Use fixed single index
                # index = 1
                # new_data[key].append(data[i][key][index])
                # if key == 'rewards':
                #   print(f"Reward of {index} segments {sum(data[i][key][index])}")
                if key == 'rewards':
                    for index in range(args.segment_len):
                        print(f"Reward of {index} segments {sum(data[i][key][index])}")
                    
        return new_data

    print(f"Combining data from {args.data_list}, segment length {args.segment_len}, total length {args.segment_len*len(args.data_list)}")
    combined_data = combine_value(data)
    # Step 3: save it in experts/ HalfCheetah-v3_Gemini.pkl
    print(f"Saving combined data to {args.save_path}")
    with open(args.save_path, 'wb') as f:
        pickle.dump(combined_data, f)

if __name__ == '__main__':
    main()