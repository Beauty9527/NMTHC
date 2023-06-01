

# sh run_seq2seq.sh long short reference
# time sh run_seq2seq.sh yeast_pac_part.fasta yeast_miseq.fasta GCF_000146045.2.fasta


long_reads = $1
short_reads = $2
reference = $3


# align and generate a complete mpileup

minimap2 --split-prefix=tmp -ax sr -t 30 $1 $2 -a --secondary=no -o short2long.bam


samtools sort short2long.bam -o short2long.sorted.bam
samtools index short2long.sorted.bam

samtools mpileup short2long.sorted.bam -a -s -f $1 -o mpileup_genome.pileup -a

#等长/itmslhppc/itmsl0212/Projects/python/wrsCode/seq2seq/data/fan/ 我们用的
python /itmslhppc/itmsl0212/Projects/python/wrsCode/seq2seqTF2.3/word_corpus_SR.py --data_dir ./ --input_file mpileup_genome.pileup --uncovered_file SR_uncovered.fasta --covered_long SR_covered_long.fasta --covered_label SR_label.fasta --long_corpus long_corpus.txt --label_corpus label_corpus.txt
python /itmslhppc/itmsl0212/Projects/python/wrsCode/seq2seqTF2.3/train_bi_lstm.py --data-dir ./ --long-corpus long_corpus.txt --label-corpus label_corpus.txt --check-point-path weight/cp.ckpt --model-hdf5-path weight/model.h5
python /itmslhppc/itmsl0212/Projects/python/wrsCode/seq2seqTF2.3/predict_bi_lstm.py --data-dir ./ --long-corpus long_corpus.txt --label-corpus label_corpus.txt --check-point-path weight/cp.ckpt --model-hdf5-path weight/model.h5 --output corrected_covered.fasta

# cat的时候未覆盖的短读必须在前
cat SR_uncovered.fasta corrected_covered.fasta > "$1"_nmt_corrected.fasta

# evaluate the results

sh evaluate.sh $3 "$1"_nmt_corrected.fasta pb

rm -rf *.bam *.fai *.bai







