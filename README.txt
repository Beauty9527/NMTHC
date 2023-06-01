#make sure that minimap2, seqtk, bedtools, samtools have been installed in your environment
conda create -n env python=3.6
conda install tensorflow-gpu2.3
sh run_nmthc.sh 
# split_big_data.py is used to sample the dataset randomly by setting the sampling ratio
# debug the program with command line.txt to see into the code.
