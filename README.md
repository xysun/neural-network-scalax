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

What's new in neural network:
- nonlinear function approximation


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