===============================
Normalising flows configuration
===============================


Flowproposal uses the implementation of normalising flow avaiable in nflows


Included normalising flows
--------------------------

FlowProposal includes three different normalising flow out-of-the-box, these
are:

- RealNVP
- MaskedAutoregressiveFlows 
- Neural Spline Flows


Using other normalising flows
-----------------------------

Other normalising flows can be implemented by the user and used with FlowProposal
by specifying the ``flow`` parameter in the ``model_config`` input dictionary.
This should be a class that either inherits from ``flowproposal.flows.Flow``:

.. literalinclude:: /../flowproposal/flows.py
   :language: python
   :lines: 122-212

