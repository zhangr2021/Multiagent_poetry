 # LLM-based multi-agent poetry generation in non-cooperative environments
(work in progress)

This repository contains the code and data for our paper: [LLM-based multi-agent poetry generation in
non-cooperative environments].

> **Abstract**: 
> Despite substantial progress of large language models (LLMs) for automatic poetry generation, the generated poetry lacks diversity while the training process differs greatly from human learning. Under the rationale that the learning process of the poetry generation systems should be more human-like and their output more diverse and novel, we introduce a framework based on social learning where we emphasize non-cooperative interactions besides cooperative interactions to encourage diversity. Our experiments are the first attempt at LLM-based multi-agent systems in non-cooperative environments for poetry generation employing both TRAINING-BASED agents (GPT-2) and PROMPTING-BASED agents (GPT-3 and GPT-4). Our evaluation based on 96k generated poems shows that our framework benefits the poetry generation process for TRAINING-BASED agents resulting in 1) a 3.0-3.7 percentage point (pp) increase in diversity and a 5.6-11.3 pp increase in novelty according to distinct and novel n-grams. The generated poetry from TRAINING-BASED agents also exhibits group divergence in terms of lexicons, styles and semantics. PROMPTING-BASED agents in our framework also benefit from non-cooperative environments and a more diverse ensemble of models with non-homogeneous agents has the potential to further enhance diversity, with an increase of 7.0-17.5 pp according to our experiments. However, PROMPTING-BASED agents show a decrease in lexical diversity over time and do not exhibit the group-based divergence intended in the social network. Our paper argues for a paradigm shift in creative tasks such as automatic poetry generation to include social learning processes (via LLM-based agent modeling) similar to human interaction.

## Generated Poetry

We release our [generated poems](dataset/poems). 

## Experiments
To reproduce the evaluations conducted in this work, please check the folder [results for section datasets](results/section_datasets) and [results for section experiments](results/section_experiments).

We provide code for both training-based [training-based](trainable_agents) and conventional frameworks  and [conversational](untrainable_agents). 

## Contacts
If you have any questions, feel free to contact us!

Ran Zhang ([ran.zhang@uni-mannheim.de](mailto:ran.zhang@uni-mannheim.de))
