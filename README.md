WIP
 
This repository will have code and slides for my upcoming ScalaX talk: neural network from scratch in scala

key components/concepts:

- fs2 for concurrent & stream; compare with vectorized Python
- "supervised learning"
- load visualise your data http://yann.lecun.com/exdb/mnist/
    - scala is pretty bad at this
    - load them as stream?
- forward propagation
- backward propagation
- cost function
- cross validation; training error, testing error
- optimizations
    - initialization
- latest technologies/algorithms
    - evolution of error rates
    - convnet

check Tensorflow interface;
implement MNIST in Tensorflow and compare speed

What's new in neural network:
- nonlinear function approximation


Computation Graph
tensorflow whitepaper: https://www.tensorflow.org/about/bib#tensorflow_a_system_for_large-scale_machine_learning
http://colah.github.io/posts/2015-08-Backprop/
http://download.tensorflow.org/paper/white_paper_tf_control_flow_implementation_2017_11_1.pdf

Backprop:
- makes training neural network feasible
- technique to calculate derivatives quickly
- derivatives are unintuitively cheap
- use State monad to store calculated derivatives
- node is definitely a container
- for graph, scala is great for data modeling

Type is great! easier to debug
Interface and inheritance is great!

Problems with scala:
- scala/jvm's image library is embarrasing; eg. no pyplot.imshow(numpy array) equivalent
- scala's file io is embarrasing, not easy to read/write to file (compared to Python's `with open()`)
- no good way to "vectorize" -- deeplearning4j

But why?
- research code is error prone (quote: "RL does not work yet", openAI paper on RND on bug)
- an ideal research language:
    - native support for parallel numerical computation on gpu
    - typed, higher kind support, basic FP
    - easy to notebook: plot, notebook should be git friendly

Goal in talk
- dymystify neural network
- show how "general" and how extensible this idea is: you can stack up arbitrary neurons and activation functions

composability: matmul + add => Linear layer; compare with Keras api https://www.tensorflow.org/tutorials/

possible questions:
- SGD
- what is the use of optimizer?

can't really shuffle

anecdote: i generalised from Double (Scalar) to Vector, then to Matrix, both succeeded in one go;
guess that's what type gives you

notice one hidden layer reduce loss very quickly

plot loss

test accuracy: 98.99%

explain why one hidden layer is worse: not enough training data (6000 data points, x parameters to tune)