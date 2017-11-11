Making model galaxies
====================================

This section of the documentation will introduce you to making model galaxies with Bagpipes. 

This is implemented through the **Model_Galaxy** class, which takes the **model_components** dictionary as an argument, into which the attributes of the model must be entered. Learn how to build a **model_components** dictionary `here <model_components.html>`_.

In order to tell the code where to find filter curves for calculating observed photometry it is necessary to define a field (so called because objects within the same fields on the sky usually have observations through the same filter curves). Learn how to do this `here <fields.html>`_.

Bagpipes is designed to rapidly update the parameters of models of the same kind, in order to faciliate fitting. Learn how to use the **update** method of **Model_Galaxy** in order to generate large numbers of models fast `here <updating.html>`_.

Finally, detailed API documentation for the **Model_Galaxy** class is available `here <api_doc.html>`_.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   model_components
   fields
   updating
   api_doc


