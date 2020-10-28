# Method

This folder contains sub class of `AbstractMethod`. Class implementing this interface allows the network to use different neetwork through the same interface.

Each method have 2 functions:
    - extract_input This method extract input used as input to the network
    - forward_policy This method use the forward pass of the network and return the output of the network (policy and value)
