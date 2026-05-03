Classifying the origin website of encrypted network traffic
is a practical privacy and security problem with applica-
tions in traffic analysis, network management, and censor-
ship research. In this project I build a supervised classifier
that predicts whether a packet capture belongs to YouTube,
Wikipedia, or Agar.io using only metadata derived from
packet headers and timing, without using plaintext payloads
or URLs.
I construct a labeled dataset from repeated visits to each
website and extract flow level and burst oriented statistics.
I compare simple baselines (linear and logistic regression)
against a gradient boosted decision tree model (LightGBM[1])
trained on an engineered feature set. Using grouped cross
validation by capture file to reduce leakage, my LightGBM
model achieves mean accuracy of 0.821 and macro F1 of 0.792.
A lightweight hyperparameter sweep improves performance
to 0.831 accuracy and 0.802 macro F1 on the same evalu-
ation protocol. I also study the effect of the flow timeout
used during feature extraction and show that timeout choice
measurably impacts downstream accuracy. The results can
be viewed at https://github.com/frigorific93/C400-traffic-analysis