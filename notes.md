## General Notes

- Most comprehensive introduction on using SAE-lens (good place to start): https://colab.research.google.com/github/jbloomAus/SAELens/blob/main/tutorials/tutorial_2_0.ipynb
- Best transformer-lens main demo: https://colab.research.google.com/github/neelnanda-io/TransformerLens/blob/main/demos/Main_Demo.ipynb#scrollTo=CYrU0hRNi7ai 
- Basic loading and analyzing SAE: https://colab.research.google.com/github/jbloomAus/SAELens/blob/main/tutorials/basic_loading_and_analysing.ipynb#scrollTo=sNSfL80Uv611 

- Sam Marks's Dictionary Learning repo, including to train different sorts of SAEs: https://github.com/saprmarks/dictionary_learning/tree/main
- Long notebook on learning to use SAEs: https://colab.research.google.com/drive/15S4ISFVMQtfc0FPi29HRaX03dWxL65zx?usp=sharing#scrollTo=0W0zgPVMw0XP

- SAE-lens documentation for potentially training or using pretrained SAEs on LLMs: https://jbloomaus.github.io/SAELens/
- Has pretrained SAEs for:
	- Mistral-7b res stream (https://huggingface.co/JoshEngels/Mistral-7B-Residual-Stream-SAEs)
	- Pythia-70b res stream (https://jbloomaus.github.io/SAELens/sae_table/#pythia-70m-deduped-res-sm)
	- Various sizes of Gemma Scope from 2b-27b (SAEs trained by Google) (https://jbloomaus.github.io/SAELens/sae_table/)
	- gpt2-small res stream


#### Gemma-scope-2-2b
Layers: 26
d_model: 2304

- Supported by transformer_lens (https://transformerlensorg.github.io/TransformerLens/index.html)
- Guide to basic use: https://colab.research.google.com/drive/17dQFYUYnuKnP6OwQPH9v_GSYUW5aj-Rp 


#### gpt2-small
Layers: 12
res_stream_dim: 768

- Supported by transformer_lens
- 