训练过程中，学习率从1×10⁻³逐步降至1×10⁻⁵。批次大小（batch size）设为 128。L2 正则项权重为4×10⁻⁵。丢弃率（dropout）设为 0.5。嵌入向量维度设为 64。

python trainDKT.py -d assist09 -m DKT

conda activate lprec
python trainSRC.py -d xes3g5m -p 3 --steps 10 --target_type portion -c 3

python trainSRC.py -d assist09 -p 3 --steps 10 --target_type all -c 3