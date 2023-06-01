"""
this file is to generate a sub file of big dataset like ecoli
choose 1/30 reads in un continues regions
每30条写出去一条
"""
import argparse


def build_parser():
    """Setup option parsing for sample."""
    parser = argparse.ArgumentParser(description='Choose 1/30 data from big dataset file like ecoli.')
    parser.add_argument('--data_dir', type=str, help='The path of datasets foder')
    parser.add_argument('--input_whole_file', type=str, help='Path of input pileup file.')
    parser.add_argument('--output_part_file', type=str, help='Path of the filtered uncovered.fasta.')
    parser.add_argument('--choose_ratio', type=int, help='how many choose one')  # 整型参数type=int
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    parsed_args = build_parser()

    data_dir = parsed_args.data_dir
    input_whole_file = data_dir + parsed_args.input_whole_file
    output_part_file = data_dir + parsed_args.output_part_file
    choose_ratio = parsed_args.choose_ratio
    with open(input_whole_file,mode='r') as whole:
        count = 0
        choose_flag = False  #
        with open(output_part_file, mode='w') as part:
            for line in whole.readlines():
                count += 1
                if line.startswith('>') and count % choose_ratio == 1:
                    choose_flag = True
                    seq_name = line
                    part.write(seq_name)
                else:
                    if choose_flag == True:
                        part.write(line)
                        choose_flag = False
    whole.close()
    part.close()
