

High level overview of the main experiment--"Testing LLM Oversight With SAEs":
1. Use open source LLMs and SAEs to find a somewhat small group (maybe <100?) of SAE neurons in each layer (or between layers?) that are most effective for recognizing a certain type of misbehavior in models. This could be anything that a deployer might reasonably want to prevent a model from doing.
2. Train classifiers on these SAE activations which detect the type of misbehavior we want to avoid
3. Report accuracy of these classifiers across different text inputs, different layers, etc.
4. Create a setup that will cut off the response if misbehavior is detected 


Things to add if we have time:
1. Compare the effectiveness of this method to a similar method based only on probing of real activations (easy)
2. Do this on multiple models, or multiple sizes of the same suite of models
3. Do this on multiple different types of misbehavior
4. See if we can design a probing setup (or something related, with SAEs) that performs better
5. Use something like "canary neurons" to update a running probability that the response is jailbroken




