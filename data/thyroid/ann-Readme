This directory contains Thyroid datasets. "ann-train.data" contains 3772 
learning examples and "ann-test.data" contains 3428 testing examples. I have 
obtained this data from Daimler-Benz. This are the informations I have got 
together with the dataset:

-------------------------------------------------------------------------------
1. Data setp summary

Number of attributes: 21 (15 attributes are binary,
			   6 attributes are continuous)
Number of classes: 3
Number of learning examples: 3772
Number of testing examples: 3428
Data set is availbale on ASCII-file

2. Description

The problem is to determine whether a patient referred to the clinic is
hypothyroid. Therefore three classes are built: normal (not hypothyroid),
hyperfunction and subnormal functioning. Because 92 percent of the patients
are not hyperthyroid a good classifier must be significant better than 92%.

Note

These are the datas Quinlans used in the case study of his article
"Simplifying Decision Trees" (International Journal of Man-Machine Studies 
(1987) 221-234)
-------------------------------------------------------------------------------


Unfortunately this data differ from the one Ross Quinlan placed in
"pub/machine-learning-databases/thyroid-disease" on "ics.uci.edu".
I don't know any more details about the dataset. But it's hard to
train Backpropagation ANNs with it. The dataset is used in two technical
reports:

-------------------------------------------------------------------------------
"Optimization of the Backpropagation Algorithm for Training Multilayer
Perceptrons":

        ftp archive.cis.ohio-state.edu  or  ftp 128.146.8.52
        cd pub/neuroprose
        binary
        get schiff.bp_speedup.ps.Z
        quit

The report is an overview of many different backprop speedup techniques.
15 different algorithms are described in detail and compared by using
a big, very hard to solve, practical data set. Learning speed and network
classification performance with respect to the training data set and also
with respect to a testing data set are discussed.
These are the tested algorithms:

backprop
backprop (batch mode)
backprop + Learning rate calculated by Eaton and Oliver's formula
backprop + decreasing learning rate (Darken and Moody)
backprop + Learning rate adaptation for each training pattern (J. Schmidhuber)
backprop + evolutionarily learning rate adaptation (R. Salomon)
backprop + angle driven learning rate adaptation(Chan and Fallside)
Polak-Ribiere + line search (Kramer and Vincentelli)
Conj. gradient + line search (Leonard and Kramer)
backprop + learning rate adaptation by sign changes (Silva and Almeida)
SuperSAB (T. Tollenaere)
Delta-Bar-Delta (Jacobs)
RPROP (Riedmiller and Braun)
Quickprop (Fahlman)
Cascade correlation (Fahlman)

-------------------------------------------------------------------------------
"Synthesis and Performance Analysis of Multilayer eural Network Architectures":


        ftp archive.cis.ohio-state.edu  or  ftp 128.146.8.52
        cd pub/neuroprose
        binary
        get schiff.gann.ps.Z
        quit

In this paper we present various approaches for automatic topology-optimization
of backpropagation networks. First of all, we review the basics of genetic
algorithms which are our essential tool for a topology search. Then we give a
survey of backprop and the topological properties of feedforward networks. We
report on pioneer work in the filed of topology--optimization. Our first
approach was based on evolutions strategies which used only mutation to change
the parent's topologies. Now, we found a way to extend this approach by an
crossover operator which is essential to all genetic search methods.
In contrast to competing approaches it allows that two parent networks with
different number of units can mate and produce a (valid) child network, which
inherits genes from both of the parents. We applied our genetic algorithm to a
medical classification problem which is extremly difficult to solve. The
performance with respect to the training set and a test set of pattern samples
was compared to fixed network topologies. Our results confirm that the topology
optimization makes sense, because the generated networks outperform the fixed
topologies and reach classification performances near optimum.

-------------------------------------------------------------------------------

Randolf Werner (evol@infko.uni-koblenz.de)

