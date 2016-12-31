## LSTM-CTC-OCR Toy experiment
The project is just a toy experiment trying to apply CTC and LSTM for OCR problem, however, I only succeed in 20-digits recognition
while longer context text is still hard to train. I may or may not pick up this project in the future. 
So basically, this is a project for summary. 

## The trend of line recognition
Recognizing lines of unconstrained text from images has always suffered from segmentation problems, which requires carefully 
designed character segmentation methods and heuristic tuning of the cost functions. However, due to the develpment of
Recurrent Neural Network, espectially LSTM(Long-Short-Term-Memory) and GRU(Gated Recurrent Unit), it is a trend to recognize
the whole line for a time and output line text from end to end.

## CTC, Connectionist Temporal Classfication
CTC, which was deviced by Alex Grave in 2006, is essentially a kind of loss function. For temporal classification tasks and
sequence labelling problems, the alignment between the inputs and outputs is unknown, so we need CTC loss function to measure
the distance between softmax activation and groundtrue label.

Baidu Research had implemented a fast parallel version of CTC, along with bindings for Torch, refer to this
[README](https://github.com/baidu-research/warp-ctc/blob/master/README.md) for more information about CTC and warp-ctc.

## Origin Reference
- Alex Graves et al. ICML 2006, [Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with Recurrent Neural Networks]
(http://www.cs.toronto.edu/~graves/icml_2006.pdf)
- Alex Graves et al. [A novel connectionist system for unconstrained handwriting recognition]
(http://www.cs.toronto.edu/~graves/tpami_2009.pdf)
- Alex Graves [A. Graves. Supervised Sequence Labelling with Recurrent Neural Networks. Textbook, Springer, 2012]
(http://www.cs.toronto.edu/~graves/preprint.pdf)

## Application of CTC
Alex Graves developed CTC and used it to speech recognition and handwriting recognition. Some researchers continued his works,
like project [ocropy](https://github.com/tmbdev/ocropy), [paragraph recognition](https://arxiv.org/abs/1604.03286), [this version]
(https://arxiv.org/abs/1604.08352), and [online seq learning](http://arxiv.org/abs/1511.06841)

You can also refer to [Recursive Recurrent Nets with Attention Modeling for OCR in the Wild]
(http://arxiv.org/abs/1603.03101) to compare these two modern different architectures.

---

You can :star: this project if you like it.

:-) 
